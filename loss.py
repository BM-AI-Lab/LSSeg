import torch
from torch import nn
from torch.nn import functional as F


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss modified in DexiNed

    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost


def bdrloss(prediction, label, radius,device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return torch.sum(cost.float().mean((1, 2, 3)))


def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss.float().mean((1, 2, 3)))


def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
    # tracingLoss

    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    prediction = torch.sigmoid(prediction)

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='none')
    cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
    label_w = (label != 0).float()
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return cost + bdr_factor * bdrcost + tex_factor * textcost


def tracing_loss(preds, labels, device):
    l_trcg = cats_loss(preds, labels, [0.01, 4.], device)
    return l_trcg


class DiceLoss(nn.Module):
    """
    Measures similarity between predicted segmentation and ground truth
    """
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # preds shape: (batch_size, num_classes, H, W)
        # targets shape: (batch_size, H, W)
        num_classes = preds.shape[1]
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # Apply softmax to preds to get probability distribution for each class
        preds_softmax = F.softmax(preds, dim=1)
        # Calculate intersection and union for each class
        intersection = torch.sum(preds_softmax * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(preds_softmax + targets_one_hot, dim=(0, 2, 3)) - intersection
        # Calculate Dice coefficient for each class
        dice_score = (2. * intersection + self.eps) / (union + self.eps)
        # Calculate Dice Loss, take average
        dice_loss = 1 - torch.mean(dice_score)
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    Addresses class imbalance issues, gives more attention to hard-to-classify samples
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(preds, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


class IoULoss(nn.Module):
    """
    Jaccard Loss
    Measures Intersection over Union (Jaccard Index) between prediction and ground truth, suitable for multi-class tasks
    """
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        labels = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()
        intersection = torch.sum(preds * labels, dim=(0, 2, 3))
        union = torch.sum(preds + labels, dim=(0, 2, 3)) - intersection
        iou = 1 - (intersection + self.eps) / (union + self.eps)
        return torch.mean(iou)



def CustomLoss(preds, labels, device):
    
    cross_entropy = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.25, gamma=2)
    iou_loss = IoULoss()

    ce = cross_entropy(preds, labels)
    dice = dice_loss(preds, labels)
    focal = focal_loss(preds, labels)
    iou = iou_loss(preds, labels)
    total_loss = ce + dice + focal + iou
    return total_loss