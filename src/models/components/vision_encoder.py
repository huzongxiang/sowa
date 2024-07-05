from typing import List, Tuple, Union
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn

from src.models.components.base import BaseEncoder


class VisionMapEncoder(BaseEncoder):
    """
    A module for encoding images using selected layers from a pretrained CLIP model and capturing their feature maps,
    and applying an projector on each captured feature map.

    Attributes:
        clip_model (nn.Module): Pretrained CLIP model.
        feature_map_idx (List[int]): Indices of the layers whose outputs are to be captured.
        layers (List[nn.Module]): List of model layers from which we will capture outputs.
        hook_handles (List[torch.utils.hooks.RemovableHandle]): List of hook handles for cleanup.
        projectors (nn.ModuleList): List of project models corresponding to each layer output.
    """
    def __init__(
        self, 
        clip_model: Union[nn.Module, partial], 
        adapter: nn.Module, 
        feature_map_idx: List[int] = [-1], 
        share_weight: bool=False
    ):
        super().__init__()
        self.clip_model = clip_model
        self.proj = clip_model.visual.proj
        self.feature_map_idx = feature_map_idx

        self.layers: List[nn.Module] = []
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.register_feature_maps()

        if isinstance(adapter, partial):
            self.adapters = nn.ModuleList([])
            for idx in feature_map_idx:
                attn = clip_model.visual.transformer.resblocks[idx].attn
                self.adapters.append(adapter(attn))
        else:
            self.adapters  = nn.ModuleList([adapter if share_weight else deepcopy(adapter) for _ in range(len(feature_map_idx))])

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Process input images through the CLIP model, retrieve feature maps, and apply alignment models.
        
        Args:
            inputs (torch.Tensor): Images to process.
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Encoded features from the CLIP model.
                - List of outputs from each alignment model corresponding to each feature map.
        """
        image_feature = self.encode(inputs)
        feature_maps = self.get_feature_maps()

        patches = [feature_map.permute(1, 0, 2)[:, 1:, :] @ self.proj for feature_map in feature_maps]
        patch_features = [adapter(patches.permute(1, 0, 2)[:, 1:, :]) @ self.proj for adapter, patches in zip(self.adapters, feature_maps)]
        # patch_features = [adapter(patches.permute(1, 0, 2)[:, 1:, :]) for adapter, patches in zip(self.adapters, feature_maps)]
        return image_feature, patches, patch_features
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the CLIP model's image encoder.
        
        Args:
            inputs (torch.Tensor): Images to encode.
        
        Returns:
            torch.Tensor: Encoded image features.
        """
        return self.clip_model.encode_image(inputs)

    def get_layer(self, idx: int) -> nn.Module:
        """
        Retrieve layer by index.
        Args:
            idx (torch.Tensor): Images to encode.
        
        Returns:
            torch.Tensor: Encoded image features.
        """
        return self.clip_model.visual.transformer.resblocks[idx]


class VisionResidualEncoder(nn.Module):
    """
    A vision encoder that uses residual connections for adaptation layers.
    
    Attributes:
        clip_model (nn.Module): The pretrained CLIP model.
        feature_map_idx (List[int]): Indices of the layers whose outputs are to be captured.
        gamma (float): Residual connection coefficient.
        adapters (nn.ModuleList): List of adapter models corresponding to each layer output.
    """
    def __init__(
        self, 
        clip_model: nn.Module, 
        adapter: nn.Module, 
        feature_map_idx: List[int] = [-1],
        share_weight: bool = False,
        gamma: float = 0.1,
    ):
        """
        Initializes the VisionResidualEncoder.
        
        Args:
            clip_model (nn.Module): The pretrained CLIP model.
            adapter (nn.Module): The adapter module to be applied to the feature maps.
            feature_map_idx (List[int]): Indices of the layers whose outputs are to be captured.
            share_weight (bool): Whether to share weights among the adapters.
            gamma (float): Residual connection coefficient.
        """
        super().__init__()
        self.clip_model = clip_model
        self.feature_map_idx = feature_map_idx
        self.gamma = gamma
        self.proj = clip_model.visual.proj

        if isinstance(adapter, partial):
            self.adapters = nn.ModuleList([])
            for idx in feature_map_idx:
                attn = clip_model.visual.transformer.resblocks[idx].attn
                self.adapters.append(adapter(attn))
        else:
            self.adapters = nn.ModuleList([adapter if share_weight else deepcopy(adapter) for _ in range(len(feature_map_idx))])

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the VisionResidualEncoder.
        
        Args:
            inputs (torch.Tensor): The input tensor.
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                - Encoded features from the CLIP model.
                - List of captured feature maps.
                - List of adapted feature maps with residual connections.
        """
        x = self.clip_model.visual.conv1(inputs)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.clip_model.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        feature_maps = []
        adapted_maps = []

        resblocks = self.clip_model.visual.transformer.resblocks

        for layer_idx in range(len(resblocks)):
            layer = resblocks[layer_idx]
            x = layer(x)
            if layer_idx in self.feature_map_idx:
                idx = self.feature_map_idx.index(layer_idx)
                feature_maps.append(x.permute(1, 0, 2)[:, 1:, :] @ self.proj)
                adapted_x = self.adapters[idx](x.permute(1, 0, 2)[:, 1:, :])
                update_x = torch.cat([x[:1, :, :], adapted_x.permute(1, 0, 2)], dim=0)
                residual = self.gamma * update_x
                x = (1 - self.gamma) * x + residual
                adapted_maps.append(adapted_x @ self.proj)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.visual.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, feature_maps, adapted_maps
    

