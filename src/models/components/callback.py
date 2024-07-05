# src/callbacks/anomaly_visualization.py
from typing import List, Any
import numpy as np
from lightning import Callback, LightningModule, Trainer

from src.utils.vision_utils import anomaly_plot
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class AnomalyVisualizationCallback(Callback):
    def __init__(self, dirpath: str, visualize: bool = False):
        """
        Callback for visualizing anomaly maps during the test phase.

        Args:
            dirpath (str): The path where the visualizations should be saved.
            visualize (bool): Whether to output the image and anormal feature map.
        """
        super().__init__()
        self.dirpath = dirpath
        self.visualize = visualize
        log.info(f"Visualize image and anormal feature map: {visualize}")

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        Called when the test batch ends. This method visualizes the anomaly maps and saves the results.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The Lightning module.
            outputs (Any): The outputs of the test step.
            batch (Any): The current batch.
            batch_idx (int): The index of the current batch.
        """
        if self.visualize:
            image_path: List[str] = batch["image_path"]
            anomaly_map: np.ndarray = outputs["anomaly_maps"]
            abnormals: np.ndarray = outputs["abnormal"]
            cls_name: List[str] = batch["cls_name"]
            anomaly_plot(image_path, anomaly_map, abnormals, self.dirpath, cls_name)
