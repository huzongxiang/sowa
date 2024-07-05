from typing import Tuple, List, Union
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.clip import clip
from src.models.components.coop import AnomalyPromptLearner
from src.models.components.text_encoder import TextMapEncoder
from src.models.components.vision_encoder import VisionMapEncoder
from src.models.components.cross_modal import FusionModule


class AnomalyCLIP(nn.Module):
    """
    A PyTorch Module for detecting anomalies using combined image and text features processed
    through a pretrained CLIP model.

    Attributes:
        clip_model (nn.Module): The loaded CLIP model.
        preprocess (Callable): Function to preprocess input images.
        text_encoder (TextMapEncoder): Encoder to process text data.
        prompt_learner (AnomalyPromptLearner): Learns and provides text prompts.
        visual_encoder (VisionMapEncoder): Encoder to process visual data.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        cfg = DotMap(**kwargs)

        (
            self.arch,
            self.image_size,
            self.class_names,
            self.temperature,
            self.prompt_length,
            self.context_length,
            self.truncate,
            self.state_template,
            self.feature_map_idx,
            self.share_weight,
            self.fusion,
            self.embedding_dim,
        ) = (
            cfg.arch,
            cfg.image_size,
            cfg.class_names,
            cfg.temperature,
            cfg.prompt_length,
            cfg.context_length,
            cfg.truncate,
            cfg.state_template,
            cfg.feature_map_idx,
            cfg.share_weight,
            cfg.fusion,
            cfg.embedding_dim,
        )

        # Initialization of CLIP components (CLIP model, preprocess) should be done here
        clip_model, _ = clip.load(cfg.arch, device="cpu")
        self.clip_model = clip_model

        # freezing backbone
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.text_encoder = TextMapEncoder(clip_model)

        self.prompt_learner = AnomalyPromptLearner(
            clip_model,
            cfg.tokenizer,
            cfg.class_names,
            cfg.prompt_length,
            cfg.context_length,
            cfg.truncate,
            cfg.state_template,
        )

        self.visual_encoder = VisionMapEncoder(
            clip_model,
            cfg.adapter,
            cfg.feature_map_idx,
            cfg.share_weight,
        )

        self.cross_fusion = FusionModule(
            cfg.fusion,
            cfg.class_names,
            cfg.embedding_dim,
        )

    def get_text_features(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generates text features from prompts.

        Returns:
            Tuple containing tensor of text features and list of tensors for additional text feature maps.
        """
        prompts, tokenized_prompts = self.prompt_learner()
        text_features, text_features_maps = self.text_encoder(prompts, tokenized_prompts)  # num_classes, 512

        return text_features, text_features_maps

    def compute_anomaly_maps(self, patch_tokens: torch.Tensor, text_features: torch.Tensor, cls_name: List[str]) -> torch.Tensor:
        """
        Computes anomaly maps using batch matrix operations from encoded patch tokens and text features.

        Args:
            patch_tokens (torch.Tensor): Encoded image patch tokens with shape [batch, layers, features].
            text_features (torch.Tensor): Encoded text features corresponding to classes, shape [classes, features].

        Returns:
            torch.Tensor: Anomaly maps after applying softmax and bilinear interpolation.
        """
        # Compute the anomaly map through batch matrix multiplication
        # anomaly_map = torch.einsum('blc,tc -> blt', patch_tokens, text_features)  # (B, L, C) @ (C, 2) -> (B, L, 2)
        anomaly_map = self.cross_fusion(text_features, patch_tokens, cls_name)               # (B, L, C) @ (C, 2) -> (B, L, 2)

        # Determine the shape for interpolation
        B, L, _ = anomaly_map.shape
        H = int((L) ** 0.5)  # Assuming a square shape for simplicity, adjust as needed

        # Reshape and interpolate to the desired image size
        anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
        anomaly_map = F.interpolate(anomaly_map, size=self.image_size, mode='bilinear', align_corners=True)

        # Apply softmax to the anomaly maps
        anomaly_map = F.softmax(anomaly_map, dim=1)    # (B, 2, H, W)
        
        return anomaly_map

    def forward(self, images: torch.Tensor, cls_name: Union[List, None]=None) -> torch.Tensor:
        """
        Forward pass of the model which computes the anomaly maps for the given input.

        Args:
            image (torch.Tensor): Input image tensor.
            cls_name (Tuple[str]): Class names associated with each image, used for retrieving text features.

        Returns:
            Tuple of tensor containing image features, tensor of text features, and list of anomaly maps.
        """
        image_features, patches, patch_feature_maps = self.visual_encoder(images)    # Placeholder: Define this function to encode images
        text_features, _ = self.get_text_features()                       # Placeholder: Define this function to get text features

        # Compute anomaly logits for each image feature
        # similarity = image_features @ text_features.t()                 # (B, C) @ (C, 2) -> (B, 2)
        similarity = self.cross_fusion(text_features, image_features, cls_name)    # (B, C) @ (C, 2) -> (B, 2)
        text_probs = F.softmax(similarity / self.temperature, dim=-1)

        # Compute anomaly maps for each feature map index
        anomaly_maps = [self.compute_anomaly_maps(patch, text_features, cls_name) for patch in patch_feature_maps]    # List[(B, 2, H, W)]

        return image_features, text_features, patches, patch_feature_maps, anomaly_maps, text_probs
