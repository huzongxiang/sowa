import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss as described in 'Focal Loss for Dense Object Detection' 
    (https://arxiv.org/abs/1708.02002), which helps focus more on hard-to-classify examples.

    Args:
        apply_nonlin (callable, optional): Non-linearity to apply to the logits.
        alpha (list, np.ndarray, float, optional): Balancing factor for each class.
        gamma (float): Focusing parameter. Higher values reduce the relative loss for 
                       well-classified examples, putting more focus on hard, misclassified examples.
        balance_index (int): Index of the class to be given more focus if alpha is set to scalar.
        smooth (float): Smoothing factor to avoid division by zero errors.
        size_average (bool): If True, returns the mean loss per batch; otherwise, returns the sum.

    Raises:
        ValueError: If `smooth` is not in the interval [0,1].
    """
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('Smooth value should be in [0,1]')

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # Flatten the logits to 2D if they are more than that (e.g., from CNNs)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.full((num_class, 1), (1 - self.alpha))
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key.scatter_(1, idx, 1)

        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma

        alpha = alpha[idx]
        alpha = alpha.squeeze()
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    """
    Implementation of the Dice Loss for binary classification problems, which is useful for
    dealing with imbalanced datasets and segmentation tasks.

    The Dice Coefficient is a measure of overlap between two samples. This loss is a measure of 
    how much the prediction differs from the ground truth in terms of the overlap.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate the number of pixels in each batch
        N = targets.size(0)
        smooth = 1  # Smoothing factor to prevent division by zero
        
        # Flatten the input and target to simplify the computation
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # Calculate the intersection and union
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        
        # Compute the Dice loss
        loss = 1 - dice_eff.sum() / N
        return loss
