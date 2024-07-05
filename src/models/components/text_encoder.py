from typing import List, Tuple
import torch
import torch.nn as nn

from src.models.components.base import BaseEncoder


class TextEncoder(nn.Module):
    """
    A module for encoding text using the text processing components of a pretrained CLIP model.
    
    Attributes:
        clip_model (nn.Module): A pretrained CLIP model containing necessary components for text encoding.
    """
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        Processes text inputs through the transformer and projects the final layer outputs.

        Args:
            prompts (torch.Tensor): Tensor of text embeddings.
            tokenized_prompts (torch.Tensor): Tensor of tokenized prompts indices for extracting the end token features.

        Returns:
            torch.Tensor: Projected features from the transformer's output corresponding to end tokens.
        """
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TextMapEncoder(BaseEncoder):
    """
    Extends the BaseEncoder to specifically handle text features extracted from a CLIP model.

    Attributes:
        clip_model (nn.Module): Pretrained CLIP model used for extracting text features.
        feature_map_idx (List[int]): List of indices specifying which transformer blocks to capture.
    """
    def __init__(self, clip_model: nn.Module, feature_map_idx: List[int] = [-1]):
        super().__init__()
        self.transformer = clip_model.transformer
        self.feature_map_idx = feature_map_idx

        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        self.layers: List[nn.Module] = []
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.register_feature_maps()

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process input prompts through the CLIP model, retrieve feature maps, and apply alignment models.
        
        Args:
            prompts (torch.Tensor): prompts to process.
            tokenized_prompts (torch.Tensor): tokenized prompts to process.
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Encoded features from the CLIP model.
                - List of outputs from each resnetblock of transformer.
        """
        cls_feature = self.encode(prompts, tokenized_prompts)    # (2 * M * K, C) for dual Coop prompt input

        feature_maps = [] 
        for x in self.get_feature_maps():
            x = x.permute(1, 0, 2)
            feature_maps.append(x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection)
        return cls_feature, feature_maps

    def encode(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        Encodes the provided prompts using the transformer model from the CLIP architecture.
        
        Args:
            prompts (torch.Tensor): Initial embeddings of the prompts.
            tokenized_prompts (torch.Tensor): Indices of tokenized prompts for final layer extraction.
        
        Returns:
            torch.Tensor: Encoded text features.
        """
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    def get_layer(self, idx: int) -> nn.Module:
        """
        Retrieve a specific transformer block by index.
        
        Args:
            idx (int): Index of the transformer block to retrieve.
        
        Returns:
            nn.Module: Transformer block at the specified index.
        """
        return self.transformer.resblocks[idx]


class TextEncoderZeroshot(nn.Module):
    """
    A zero-shot learning approach for text encoding using a CLIP model without any fine-tuning or training.
    
    Attributes:
        clip_model (nn.Module): A pretrained CLIP model containing necessary components for text encoding.
    """
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the zero-shot text encoder.

        Args:
            text (torch.Tensor): Input text to encode.

        Returns:
            torch.Tensor: Encoded text features.
        """
        x = self.token_embedding(text) # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # Extract features from the end of token embedding
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
