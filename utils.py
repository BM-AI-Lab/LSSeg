import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import os
import yaml
import numpy as np
import pandas as pd
import re


class jindu():
    """ 
    >>> import time
    >>> bar = jindu(100, name='Bar', limit_length=30)    # Define progress bar
    >>> for i in range(100):
    >>>     time.sleep(0.1) 
    >>>     if i % 20 == 0:
    >>>         bar.show(f'{i}+{i}={i + i}')             # Other output in code
    >>>     bar.update()                                 # Update progress bar
    0+0=0
    20+20=40
    40+40=80
    60+60=120
    80+80=160
    Bar: ==============================> 100.0%
    """
    def __init__(self, total, name='', limit_length=None):
        """ total: Total number of items
        name: Name of the progress bar
        limit_length: Display length of the progress bar
        """
        self.name = name
        self.real_now = 0
        self.now = 0
        self.total_length = total
        self.limit_length = limit_length if limit_length else total
    
    def before_print(self):
        """ Clear line before output
        """
        if self.name != '':
            print(' ' * (len(self.name) + 2), end='')
        print(' ' * (self.limit_length + 6), end='\r')

    def update(self, i=1):
        """ Update progress bar
        """
        self.real_now += i
        self.now = int(self.real_now / self.total_length * self.limit_length)
        if self.name != '':
            print(self.name + ':', end=' ')
        print('=' * self.now + '>' + '-' * (self.limit_length - 1 - self.now), 
              f'{(self.real_now / self.total_length * 100.):.1f}%',
              end='\r' if self.real_now < self.total_length else '\n')

    def show(self, *args, **kwargs):
        """ Usage same as print function
        """
        self.before_print()
        for item in args:
            print(item, end=' ')
        print(end=kwargs['end'] if kwargs else '\n')


class Smish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh(torch.log(1+torch.sigmoid(input)))


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def grayscale_linear_transform(images, eps=1e-12):
    """ Grayscale linear transformation
    """
    return (images - images.min()) / (images.max() - images.min() + eps)


def show_result(folder, img_lst, num_rows, num_cols, titles, scale=5):
    """
    img_lst[0]: the raw RGB image. shape: 3xHxW
    img_lst[1]: the label.         shape: 1xHxW
    -                              shape: 1xHxW
    """
    assert num_rows * num_cols >= len(img_lst)

    raw, label = img_lst[:2]
    if raw.shape[0] == 1:
        raw =  torch.cat([raw, raw, raw], dim=0).permute(1, 2, 0)
    else:
        raw = raw.permute(1, 2, 0)
    label = torch.cat([label, label, label], dim=0).permute(1, 2, 0)

    ## 1xHxW -> HxWx3
    preds = [torch.cat([pred] * 3, dim=0).permute(1, 2, 0) for pred in img_lst[2:]]

    d2l.show_images([raw, label] + preds, num_rows=num_rows, num_cols=num_cols, titles=titles, scale=scale)
    os.makedirs(folder, exist_ok=True)
    d2l.plt.savefig(os.path.join(folder, 'show_result.png'), bbox_inches='tight')
    d2l.plt.close()


def aggregate_metrics(root, output_file):
    # Store all metrics data
    all_metrics = []

    # Iterate through each folder
    for folder in os.listdir(root):
        # Build path to metrics.csv file
        csv_path = os.path.join(root, folder, "metrics.csv")
        print(csv_path)

        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_path} does not exist, skipping this folder.")
            continue

        # Read CSV file
        try:
            # Read CSV file, skip first line as column names
            df = pd.read_csv(csv_path, header=0)
            # Convert values to float and store
            all_metrics.append(df)
        except Exception as e:
            print(f"Error reading file {csv_path}: {e}")
            continue

    # Combine all metrics
    if not all_metrics:
        print("No valid metrics data found, cannot create output file.")
        return

    # Combine data from all files
    combined_metrics = pd.concat(all_metrics, axis=0)
    
    # Directly extract all values
    all_values = []
    for col in combined_metrics.columns:
        # Get column data
        col_data = combined_metrics[col].to_numpy()
        all_values.append(col_data)

    # Calculate average for each column
    mean_values = [np.mean(vals) for vals in all_values]

    # Create result Series
    result = pd.Series(mean_values, index=df.columns)

    pd.DataFrame([result]).to_csv(output_file, index=False, encoding='utf-8')



def load_config(path):
    """ Load configuration """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config):
    """ Save configuration """
    with open(os.path.join(config['save_path'], 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    return input * torch.tanh(torch.log(1+torch.sigmoid(input)))


def nms_for_edge_confidence_batch(edge_confidence_batch, kernel_size=3):
    """
    Apply Non-Maximum Suppression (NMS) to batch of edge confidence images

    Args:
        edge_confidence_batch (torch.Tensor): Input batch of edge confidence images, shape (batch_size, height, width)
        kernel_size (int): Neighborhood size, typically 3 or 5

    Returns:
        torch.Tensor: Batch of edge confidence images after NMS processing
    """
    # Ensure input is torch.Tensor
    edge_confidence_batch = torch.as_tensor(edge_confidence_batch)
    batch_size, height, width = edge_confidence_batch.shape

    # Create an output image of the same size as input, initialized to 0
    output_batch = torch.zeros_like(edge_confidence_batch)

    # Calculate neighborhood radius
    pad_size = kernel_size // 2

    # Process each sample
    for b in range(batch_size):
        # Get the edge confidence image of current sample
        current_image = edge_confidence_batch[b]

        # Create an output image of the same size as current image, initialized to 0
        output_image = torch.zeros_like(current_image)

        # Use replicate_pad to pad the image to handle boundary cases
        padded_image = torch.nn.functional.pad(current_image.unsqueeze(0).unsqueeze(0), 
                                             [pad_size, pad_size, pad_size, pad_size], 
                                             mode='replicate').squeeze()

        # Iterate through each pixel in the current image
        for i in range(height):
            for j in range(width):
                # Extract current pixel and its neighborhood
                current_value = current_image[i, j].item()
                neighbor_values = padded_image[i:i+kernel_size, j:j+kernel_size]

                # Find the maximum value in the neighborhood
                max_value = neighbor_values.max().item()

                # If current pixel is the maximum in the neighborhood, keep it; otherwise, set to 0
                if current_value == max_value:
                    output_image[i, j] = current_value
                else:
                    output_image[i, j] = 0

        # Add the processed image to the output batch
        output_batch[b] = output_image

    return output_batch


def iterative_thresholding_torch(image_batch, max_iterations=100, tolerance=1e-3):
    """
    Iterative thresholding segmentation function (PyTorch-based), supports batch processing

    Args:
        image_batch (torch.Tensor): Input image tensor, shape (B, 1, H, W)
        max_iterations (int): Maximum number of iterations
        tolerance (float): Threshold change tolerance

    Returns:
        tuple: Segmented image tensor (0 and 255 values), final threshold for each image
    """
    batch_size, channels, height, width = image_batch.shape
    device = image_batch.device

    # Apply Gaussian smoothing
    kernel_size = 5
    sigma = 2.0
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]  # Create 2D Gaussian kernel
    kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Adapt kernel shape for convolution operation

    # Apply Gaussian smoothing to the image
    padding = kernel_size // 2
    smoothed_image_batch = F.conv2d(image_batch, kernel, padding=padding, groups=1)

    # Initialize threshold (can use image mean as initial threshold)
    thresholds = smoothed_image_batch.mean(dim=(1, 2, 3))  # Calculate initial threshold for each image

    for _ in range(max_iterations):
        new_thresholds = torch.zeros_like(thresholds)
        for i in range(batch_size):
            smoothed_image = smoothed_image_batch[i, 0]  # Current image
            threshold = thresholds[i]

            # Segment image into foreground and background based on current threshold
            foreground_mask = smoothed_image >= threshold
            background_mask = smoothed_image < threshold

            # Calculate average gray values of foreground and background
            mean_foreground = smoothed_image[foreground_mask].mean() if foreground_mask.any() else threshold
            mean_background = smoothed_image[background_mask].mean() if background_mask.any() else threshold

            # Calculate new threshold
            new_threshold = (mean_foreground + mean_background) / 2.0
            new_thresholds[i] = new_threshold

        # Check if threshold change is less than tolerance
        changes = torch.abs(new_thresholds - thresholds)
        if torch.all(changes < tolerance):
            break
        thresholds = new_thresholds

    # Use final threshold to suppress noise
    segmented_image_batch = torch.zeros_like(smoothed_image_batch)
    for i in range(batch_size):
        segmented_image_batch[i, 0] = torch.where(smoothed_image_batch[i, 0] >= thresholds[i], 
                                                  smoothed_image_batch[i, 0], 
                                                  torch.tensor(0.0, device=device))

    return segmented_image_batch