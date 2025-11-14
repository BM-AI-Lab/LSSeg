import torch
from torch.nn import functional as F
import os
import torchmetrics
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import MulticlassJaccardIndex, AveragePrecision
from torchmetrics.image import PeakSignalNoiseRatio
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.morphology import skeletonize
from thop import profile

from dataset import SegImageDataset
from utils import load_config, grayscale_linear_transform, nms_for_edge_confidence_batch, iterative_thresholding_torch
from models.lsseg import LSSeg


class OIS(Metric):
    def __init__(self, thresholds=100, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = torch.linspace(0, 1, thresholds)
        
        self.add_state("total_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (N, 1, H, W) tensor with edge probability predictions
            target: (N, 1, H, W) tensor with binary edge annotations
        """
        # Move thresholds to same device as preds
        thresholds = self.thresholds.to(preds.device)
        
        target = target.bool()  # (N, 1, H, W)
        
        # Calculate binary masks for all thresholds
        binary_masks = preds > thresholds.view(1, -1, 1, 1)  # (N, T, H, W)
        
        # Calculate TP, FP, FN
        tp = (binary_masks & target).sum(dim=(2, 3))  # (N, T)
        fp = (binary_masks & ~target).sum(dim=(2, 3))
        fn = (~binary_masks & target).sum(dim=(2, 3))
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Find best F1 for each image
        best_f1 = f1.max(dim=1).values  # (N,)
        
        # Update states
        self.total_f1 += best_f1.sum()
        self.total_samples += preds.shape[0]

    def compute(self):
        return self.total_f1 / self.total_samples


class ODS(Metric):
    def __init__(self, thresholds=100, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = torch.linspace(0, 1, thresholds)
        
        self.add_state("tp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (N, 1, H, W) tensor with edge probability predictions
            target: (N, 1, H, W) tensor with binary edge annotations
        """
        thresholds = self.thresholds.to(preds.device)
        
        target = target.bool()  # (N, 1, H, W)
        
        # Calculate binary masks
        binary_masks = preds > thresholds.view(1, -1, 1, 1)  # (N, T, H, W)
        
        # Calculate TP, FP, FN
        tp = (binary_masks & target).sum(dim=(0, 2, 3))  # (T,)
        fp = (binary_masks & ~target).sum(dim=(0, 2, 3))
        fn = (~binary_masks & target).sum(dim=(0, 2, 3))
        
        # Update states
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        return f1_scores.max()


class Dice(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        # Initialize state to store cumulative values during calculation
        self.add_state("true_positive", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Flatten predictions and targets
        preds = preds.flatten()
        target = target.flatten()

        # Convert predictions to binary values based on threshold
        preds = (preds > self.threshold).float()

        # Calculate relevant statistics
        self.true_positive += torch.sum(preds * target).item()
        self.false_positive += torch.sum(preds * (1 - target)).item()
        self.false_negative += torch.sum((1 - preds) * target).item()

    def compute(self):
        # Calculate Dice coefficient
        numerator = 2 * self.true_positive
        denominator = numerator + self.false_positive + self.false_negative

        # Avoid division by zero
        dice = numerator / (denominator + 1e-8)
        return dice


class DiceIoUWithNMS(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        # Initialize state to store cumulative values during calculation
        self.add_state("true_positive", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Flatten predictions and targets
        preds = preds.flatten()
        target = target.flatten()
        preds_after_nms = preds

        # Calculate relevant statistics
        self.true_positive += torch.sum(preds_after_nms * target).item()
        self.false_positive += torch.sum(preds_after_nms * (1 - target)).item()
        self.false_negative += torch.sum((1 - preds_after_nms) * target).item()

        # Calculate IoU-related statistics
        self.intersection += torch.sum(preds_after_nms * target).item()
        self.union += torch.sum(preds_after_nms + target - preds_after_nms * target).item()

    def compute(self):
        # Calculate Dice coefficient
        numerator = 2 * self.true_positive
        denominator = numerator + self.false_positive + self.false_negative

        # Avoid division by zero
        dice = numerator / (denominator + 1e-8)

        # Calculate IoU
        iou = self.intersection / (self.union + 1e-8)

        return {"dice": dice, "iou": iou}


def count(model, device, config):
    input = torch.randn(1, config['in_channels'], 512, 512).to(device)  # input shape (batch_size, channels, height, width)

    # Calculate FLOPs and parameter count
    flops, params = profile(model, inputs=(input,))

    return {'FLOPs': flops, 'Params': params}


def evaluate(model, data_iter, device, config):

    metrics = MetricCollection(
        {
            'ois': OIS(),
            'ods': ODS(),
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'dice_iou': DiceIoUWithNMS(threshold=0.19),   # 0.19 best for LTGNet Vessel
            'psnr': PeakSignalNoiseRatio()
        }
    ).to(device)

    model.to(device)
    model.eval()

    for images, labels in tqdm(data_iter):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)

        preds = iterative_thresholding_torch(grayscale_linear_transform(F.sigmoid(preds)))

        metrics.update(preds, labels)
    
    metric = {key: value.detach().cpu().numpy() for key, value in metrics.compute().items()}
    return metric | count(model, device, config)


if __name__ == '__main__':
    
    model_path = 'log/LSSeg188_tubulin/fold_0/best.params'
    save_path = os.path.dirname(model_path)

    config = load_config(os.path.join(save_path, 'config.yaml'))

    model = LSSeg(in_channels=config['in_channels'])
    model.load_state_dict(torch.load(model_path))

    image_path, label_path = pd.read_csv(config['data_pair_file'], header=None).values.T
    data_iter = DataLoader(SegImageDataset(image_path, label_path, config['resize'], config['image_mode']),
                           batch_size=config['batch_size'], shuffle=False, drop_last=False)
    
    metrics = evaluate(model, data_iter, device=torch.device('cuda'))
    print(metrics)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_path, 'metrics.csv'), index=False, encoding='utf-8')