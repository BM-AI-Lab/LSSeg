from PIL import Image
import os
import json
import torchvision
from torch.utils.data import Dataset, DataLoader


class SegImageDataset(Dataset):
    """ Segmentation image loading dataset """
    def __init__(self, image_path, label_path, resize=(512, 512), image_mode='RGB'):
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.transform_norm = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        self.image_mode = image_mode

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert(self.image_mode)
        if self.image_mode == 'RGB':
            image = self.transform_norm(image)
        else:
            image = self.to_tensor(image)
        
        label = Image.open(self.label_path[index]).convert('L')
        label = self.to_tensor(label)

        return image, label

    @staticmethod
    def get_dataloader(train_image_path, train_label_path, test_image_path, test_label_path, config):
        train_dataset = SegImageDataset(train_image_path, train_label_path, config['resize'], config['image_mode'])
        test_dataset = SegImageDataset(test_image_path, test_label_path, config['resize'], config['image_mode'])
        return (DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=config['drop_last']), 
                DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=config['drop_last']))

