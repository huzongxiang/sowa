from typing import List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from src.models.components.swin_transformer.swin_transformer_v2 import window_partition, window_reverse


class Linear(nn.Module):
    """
    Multi-layer Perceptron (MLP) with customizable layers and activation function.
    
    Args:
        in_features (int): Dimension of the input features.
        out_features (int): Dimension of the output features.
        hidden_features (List[int]): List of dimensions for each hidden layer.
        activation (nn.Module): The activation function to use after each hidden layer.
        dropout_prob (float): Dropout probability. Default: 0.0.
        bias (bool): If set to False, the layers will not learn an additive bias. Default: False.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        hidden_features: Union[List, None] = None, 
        activation: nn.Module = nn.ReLU(),
        dropout_prob: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_features = hidden_features if hidden_features is not None else []
        self.activation = activation
        self.dropout_prob = dropout_prob
        
        # Create the layers based on the hidden dimensions
        layers = []
        last_dim = in_features
        
        for dim in self.hidden_features:
            layers.append(nn.Linear(last_dim, dim, bias))
            layers.append(activation)
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))
            last_dim = dim
        
        # Add the final layer
        layers.append(nn.Linear(last_dim, out_features, bias))
        
        # If there's only one layer, it will be just a Linear layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor of the network.
        """
        return self.layers(x)


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted windows.

    Args:
        attn (nn.Module): Pre-trained attention module from which to clone weights.
        window_size (tuple[int, int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        cpb_dim (int, optional): Hidden dimension for the relative position bias MLP. Default: 64
        value_only (bool, optional): If True, only use value vectors in attention calculation. Default: False
        attn_drop (float, optional): Dropout ratio of attention weights. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, attn: nn.Module, window_size: tuple[int, int], num_heads: int, cpb_dim: int = 64, 
                 value_only: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        self.value_only = value_only

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(cpb_dim, num_heads, bias=False)
        )

        # Get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # Shape: (1, 2*Wh-1, 2*Ww-1, 2)

        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # Normalize to [-8, 8]
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # Shape: (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # Shape: (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # Shape: (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Shape: (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Shape: (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = attn.in_proj_weight
        self.qkv.requires_grad = False

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = attn.out_proj
        self.proj.requires_grad = False

        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features with shape (num_windows*B, N, C)
            mask (torch.Tensor, optional): Mask with shape (num_windows, Wh*Ww, Wh*Ww) or None. Default: None
        """
        B_, N, C = x.shape

        qkv = F.linear(input=x, weight=self.qkv)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

        # Cosine attention
        if not self.value_only:
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        else:
            attn = (F.normalize(v, dim=-1) @ F.normalize(v, dim=-1).transpose(-2, -1))
            
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Shape: (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # Shape: (nH, Wh*Ww, Wh*Ww)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        attn (nn.Module): Pre-trained attention module from which to clone weights.
        input_resolution (tuple[int, int]): Input resolution (H, W).
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int, optional): Shift size for SW-MSA. Default: 0.
        cpb_dim (int, optional): Hidden dimension for the relative position bias MLP. Default: 64.
        value_only (bool, optional): If True, only use value vectors in attention calculation. Default: False.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, attn: nn.Module, input_resolution: tuple[int, int], window_size: int, num_heads: int, 
                 shift_size: int = 0, out_features: int =768, cpb_dim: int = 64, value_only: bool = False,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.GELU(), hidden_features: Union[List, None] = None):
        super().__init__()
        dim = attn.in_proj_weight.shape[-1]
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
             attn, window_size=to_2tuple(self.window_size), num_heads=num_heads,
             cpb_dim=cpb_dim, value_only=value_only, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(out_features)
        self.mlp = Linear(in_features=dim, out_features=out_features, hidden_features=hidden_features, activation=act_layer, dropout_prob=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # Shape: (1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # Shape: (nW, window_size, window_size, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # Shape: (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # Shape: (nW*B, window_size*window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # Shape: (nW*B, window_size*window_size, C)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # Shape: (B, H', W', C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        attn (nn.Module): Pre-trained attention module from which to clone weights.
        input_resolution (tuple[int, int]): Input resolution (H, W).
        window_size (int): Local window size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        cpb_dim (int, optional): Hidden dimension for the relative position bias MLP. Default: 64.
        value_only (bool, optional): If True, only use value vectors in attention calculation. Default: False.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, attn: nn.Module, input_resolution: tuple[int, int], window_size: int, depth: int, num_heads: int,
                 out_features: int =1024, cpb_dim: int = 64, value_only: bool = False, drop: float = 0., attn_drop: float = 0.,
                 drop_path: Union[float, tuple[float]] = 0., norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.GELU(), hidden_features: Union[List, None] = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(attn=attn, input_resolution=input_resolution, 
                                 window_size=window_size, num_heads=num_heads, 
                                 out_features=out_features,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 cpb_dim=cpb_dim, value_only=value_only,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 hidden_features=hidden_features)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
