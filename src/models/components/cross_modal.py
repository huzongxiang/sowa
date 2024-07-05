from typing import Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalModule(nn.Module):
    """
    A module for fusing text and vision features using various methods.
    
    Args:
        fusion_method (nn.Module): The fusion method to use.
        class_names (List[str]): List of class names.
        embedding_dim (int): The dimension of the embeddings.
        learnable (bool): Whether to use a learnable transformation for the text features.
    """
    def __init__(self, fusion_method: nn.Module, embedding_dim: Union[None, int] = None):
        super().__init__()
        self.fusion_method = fusion_method
        self.embedding_dim = embedding_dim

        if self.embedding_dim is not None:
            self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion module.

        Args:
            text_features: Tensor of shape (B, 2, C), where B is the batchsize, 2 is the number of states, C is the embedding dimension.
            vision_features: Tensor of shape (B, L, C) for patch tokens or (B, C) for cls token.
        Returns:
            Fused features of shape (B, num_classes) for classification or (B, L, num_classes) for anomaly maps.
        """
        text_features = text_features.to(vision_features.device)
        if self.embedding_dim is not None:
            text_features = self.linear(text_features)

        return self.fusion_method(text_features, vision_features)


class DotProductFusion(nn.Module):
    """
    A class for performing dot product fusion of text and vision features.
    """

    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Perform dot product fusion of text and vision features.

        Args:
            text_features: Tensor of shape (B, 2, C).
            vision_features: Tensor of shape (B, L, C) for patch tokens or (B, C) for cls token.

        Returns:
            Fused features of shape (B, 2) or (B, L, 2).
        """
        text_features = F.normalize(text_features, p=2, dim=-1)
        vision_features = F.normalize(vision_features, p=2, dim=-1)

        if vision_features.dim() == 3:  # Patch tokens
            return torch.einsum('blc,bkc -> blk', vision_features, text_features)  # (B, L, 2)
        else:  # CLS token
            return (vision_features.unsqueeze(1) @ text_features.transpose(1, 2)).squeeze(1)  # (B, 2)


class FusionModule(nn.Module):
    """
    A module for fusing text and vision features using various methods.
    
    Args:
        fusion_method (nn.Module): The fusion method to use.
        class_names (List[str]): List of class names.
        embedding_dim (int): The dimension of the embeddings.
        learnable (bool): Whether to use a learnable transformation for the text features.
    """
    def __init__(self, fusion_method: nn.Module, class_names: List[str], embedding_dim: Union[None, int] = None):
        super().__init__()
        self.fusion_method = fusion_method
        self.class_names = class_names
        self.embedding_dim = embedding_dim

        # Create a mapping from class names to indices
        self.class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}

        if self.embedding_dim is not None:
            self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor, cls_names: List[str]) -> torch.Tensor:
        """
        Forward pass of the fusion module.

        Args:
            text_features: Tensor of shape (M * K * 2, C), where M is the number of classes, K is the number of templates, C is the embedding dimension.
            vision_features: Tensor of shape (B, L, C) for patch tokens or (B, C) for cls token.
            cls_names: List of class names for corresponding features.

        Returns:
            Fused features of shape (B, num_classes) for classification or (B, L, num_classes) for anomaly maps.
        """
        pooled_text_features = self.pool_text_features(text_features)  # Pool to (M, 2, C)
        batch_text_features = self.get_batch_text_features(pooled_text_features, cls_names)  # (B, 2, C)

        return self.fusion_method(batch_text_features, vision_features)

    def pool_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Pool text features within normal and anomaly categories to get shape (M, 2, C).

        Args:
            text_features: Tensor of shape (2 * M * K, C).

        Returns:
            Pooled text features of shape (M, 2, C).
        """
        if self.embedding_dim is not None:
            text_features = self.linear(text_features)

        M = len(self.class_names)
        K = text_features.size(0) // (2 * M)

        pooled_text_features = text_features.view(2, M, K, -1).mean(dim=2).permute(1, 0, 2)  # (M, 2, C)
        return pooled_text_features

    def get_batch_text_features(self, pooled_text_features: torch.Tensor, cls_names: List[str]) -> torch.Tensor:
        """
        Get text features for a batch of class names.

        Args:
            pooled_text_features: Tensor of shape (M, 2, C).
            cls_names: List of class names for corresponding features.

        Returns:
            Batch text features of shape (B, 2, C).
        """
        cls_indices = [self.class_name_to_idx.get(cls_name, 0) for cls_name in cls_names]
        batch_text_features = pooled_text_features[cls_indices]    # (B, 2, C)
        return batch_text_features