import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split
from d2l import torch as d2l

from models.model import TImE
from models.dense_time import DenseTImE
from models.teed import TEED
# from models.line_net import LineNet
from models.line_net2 import LineNet
from dataset import get_dataloader, SegImageDataset
from loss import tracing_loss, CustomLoss
from utils import grayscale_linear_transform, show_result, load_config, save_config, iterative_thresholding_torch
from utils import aggregate_metrics
from eval import evaluate
from models.lsseg import LSSeg


def test(model, test_iter, loss, device, config):
    """ Model loss on test set
    """
    model.eval()
    ls = []
    for images, labels in test_iter:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        l = loss(preds, labels, device)
        ls.append(l.item())
    # Randomly show results
    idx = random.randint(0, images.shape[0] - 1)    # Randomly select an image from the batch
    img_lst = [images.cpu().detach()[idx],
                1 - labels.cpu().detach()[idx],
                1 - iterative_thresholding_torch(grayscale_linear_transform(F.sigmoid(preds.cpu().detach())))[idx]]
    show_result(config['save_path'], img_lst, num_rows=1, num_cols=3, titles=['Raw Image', 'Ground Truth', 'Predict'])
    return np.array(ls).mean()


def train_one_epoch(model, train_iter, optimizer, loss, device):
    """ Train for one epoch """
    model.train()
    ls = []
    for images, labels in train_iter:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)

        l = loss(preds, labels, device)
        l.backward()
        optimizer.step()
        
        ls.append(l.item())
    return np.array(ls).mean()


def train(model, train_iter, test_iter, loss, device, config):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config['num_epochs'])
    animator = d2l.Animator(xlabel='Epochs', ylabel='Loss', xlim=[1, config['num_epochs']],
                            legend=['Train Loss', 'Test Loss'], figsize=(7, 5))

    best_test_l, best_epoch = None, None

    for epoch in range(config['num_epochs']):
        train_l = train_one_epoch(model, train_iter, optimizer, loss, device)
        test_l = test(model, test_iter, loss, device, config)
        
        # Save weights
        if best_epoch == None or test_l < best_test_l:
            best_test_l = test_l
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config['save_path'], 'best.params'))

        animator.add(epoch + 1, (train_l, test_l))
        d2l.plt.savefig(os.path.join(config['save_path'], 'curve.png'), dpi=300)

        # Output current epoch results
        print(f'epoch {epoch}:', f'loss {train_l:.5f}, test loss {test_l:.5f}')

        scheduler.step()
    
    # Save config
    save_config(config)


def five_fold_validation(model_name, loss, device, config, desc):
    """ 5-fold cross validation """
    image_path, label_path = pd.read_csv(config['data_pair_file'], header=None).values.T
    kf = KFold(n_splits=5, shuffle=True, random_state=33)
    for i, (train_idx, test_idx) in enumerate(kf.split(image_path)):
        train_image_path, test_image_path = image_path[train_idx], image_path[test_idx]
        train_label_path, test_label_path = label_path[train_idx], label_path[test_idx]
        train_iter, test_iter = SegImageDataset.get_dataloader(train_image_path, train_label_path,
                                                               test_image_path, test_label_path, config)
        model = model_name(in_channels=config['in_channels']).to(device)
        print(desc, f'Fold {i} training')
        save_path = os.path.join('log', desc, f'fold_{i}')
        os.makedirs(save_path, exist_ok=True)
        config['save_path'] = save_path
        train(model, train_iter, test_iter, loss, device, config)
        
        # Evaluate model
        model.load_state_dict(torch.load(os.path.join(config['save_path'], 'best.params')))
        metrics = evaluate(model, test_iter, device=device, config=config)
        print(desc, f'Fold {i}', metrics)
        pd.DataFrame([metrics]).to_csv(os.path.join(save_path, 'metrics.csv'), index=False, encoding='utf-8')
    
    # Calculate average metrics
    aggregate_metrics(os.path.join('log', desc), os.path.join('log', desc, 'mean_metrics.csv'))


def loss(preds, labels, device):
    """
    loss for every level
    """
    l = 0
    for p in preds:
        l += tracing_loss(p, labels, device)
    return l


if __name__ == '__main__':

    # Load configuration
    config = load_config('config.yaml')

    model_name = LSSeg
    loss = tracing_loss
    device = d2l.try_gpu()

    desc = input('log :')

    five_fold_validation(model_name, loss, device, config, desc)
