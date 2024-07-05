from typing import List, Tuple
from copy import deepcopy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Hook to be attached to a model layer.
    
    Args:
        module (nn.Module): The layer to which the hook is registered.
        input (torch.Tensor): The input to the layer.
        output (torch.Tensor): The output from the layer.

    This hook saves the output of the layer to an attribute of the module.
    """
    module.output = output


class BaseEncoder(nn.Module, ABC):
    """
    A general module for encoding inputs (either images or text) using selected layers from a pretrained model
    and capturing their feature maps.
    """

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs using the model's specific encoder.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_layer(self, idx: int) -> nn.Module:
        """
        Retrieve layer by index.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process inputs through the model, retrieve feature maps, and apply alignment models.
        """
        pass

    def register_feature_maps(self) -> None:
        """
        Register hooks on specified layers to capture their outputs.
        """
        for idx in self.feature_map_idx:
            layer = self.get_layer(idx)
            hook_handle = layer.register_forward_hook(hook)
            self.layers.append(layer)
            self.hook_handles.append(hook_handle)

    def get_feature_maps(self) -> List[torch.Tensor]:
        """
        Retrieve captured feature maps from the selected layers.
        """
        return [layer.output for layer in self.layers if hasattr(layer, 'output')]

    def remove_handles(self) -> None:
        """
        Remove all registered hooks to clean up resources.
        """
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []
        self.layers = []

    def __del__(self):
        """
        Ensure all hooks are removed when the instance is deleted.
        """
        self.remove_handles()
