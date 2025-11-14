import os
import cv2
from PIL import Image
import torch
import torchvision
from torch.nn import functional as F

from utils import grayscale_linear_transform, iterative_thresholding_torch
from models.lsseg import LSSeg


def load_images(image_folder, mode):
    to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.ToTensor()
        ])
    
    images = []
    for img in sorted(os.listdir(image_folder)):
        print(img)
        image = Image.open(os.path.join(image_folder, img)).convert(mode)
        images.append(to_tensor(image))
    
    return torch.stack(images, dim=0)


def predict(X, model, checkpoint_path, device, is_lst=False):
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    preds = model(X.to(device))[-1] if is_lst else model(X.to(device))
    preds = 1 - iterative_thresholding_torch(grayscale_linear_transform(F.sigmoid(preds)))
    return preds.cpu()


def save_images(images, folder):
    os.makedirs(folder, exist_ok=True)

    for i, image in enumerate(images.detach()):
        image = torch.cat([image, image, image], dim=0)
        image = image.permute(1, 2, 0).numpy() * 255
        
        cv2.imwrite(os.path.join(folder, f'{i + 1}.png'), image)


if __name__ == '__main__':

    device = torch.device('cpu')

    model = LSSeg(in_channels=[1, 8, 8])
    
    # Model weights
    checkpoint_path = 'log/LSSeg188_tubulin/fold_0/best.params'

    # According to data type: tubulin & muscle:L   vessels: RGB
    images = load_images(image_folder='data/data_sample/tubulin/Input_image', mode='L')
    
    # is_lst indicates whether model output is a list
    preds = predict(images, model, checkpoint_path, device)

    # Save results
    save_images(preds, folder=f'data/data_sample/tubulin/LSSeg')