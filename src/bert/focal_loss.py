import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha=1, gamma=2):
    """
    Focal Loss for multilabel classification.
    
    Args:
        inputs (torch.Tensor): Predictions (logits) of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth binary labels of shape (batch_size, num_classes).
        alpha (float): Weighting factor for balancing the importance of positive/negative examples.
        gamma (float): Focusing parameter for modulating the loss.

    Returns:
        torch.Tensor: Focal loss value.
    """
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # pt is the probability of being classified correctly
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()