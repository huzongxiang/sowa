import torch
import torch.nn.functional as F


def cosine_similarity_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity between rows of X and rows of Y using PyTorch.

    Args:
        X (torch.Tensor): Tensor of shape (n_samples_X, n_features)
        Y (torch.Tensor): Tensor of shape (n_samples_Y, n_features)

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (n_samples_X, n_samples_Y)
    """
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    return torch.mm(X_norm, Y_norm.t())