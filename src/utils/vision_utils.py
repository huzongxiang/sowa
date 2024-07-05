from typing import List, Optional
import cv2
import numpy as np
from pathlib import Path


def normalize(pred: np.ndarray, max_value: Optional[float] = None, min_value: Optional[float] = None) -> np.ndarray:
    """
    Normalize the prediction values to a range between 0 and 1.

    Args:
        pred (np.ndarray): The array to be normalized.
        max_value (Optional[float]): The maximum value for normalization. If None, the max of pred is used.
        min_value (Optional[float]): The minimum value for normalization. If None, the min of pred is used.

    Returns:
        np.ndarray: The normalized array.
    """
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())

    return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image: np.ndarray, scoremap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply the anomaly detection score map to the image with an alpha blending.

    Args:
        image (np.ndarray): The original image in RGB format.
        scoremap (np.ndarray): The anomaly score map, assumed to be normalized between 0 and 1.
        alpha (float): The blending factor between the image and the score map.

    Returns:
        np.ndarray: The resulting image with the score map applied.
    """
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def anomaly_plot(paths: List[str], anomaly_map: np.ndarray, abnormals: np.ndarray, save_path: str, cls_name: List[str]) -> None:
    """
    Visualize anomaly maps overlaid on the original images and save the visualizations.

    Args:
        paths (List[str]): List of paths to the original images.
        anomaly_map (np.ndarray): Array of anomaly maps corresponding to the images.
        save_path (str): The path where the visualizations should be saved.
        cls_name (List[str]): List of class names corresponding to each image.
    """
    img_size = anomaly_map.shape[-1]
    anomaly_map = anomaly_map.detach().cpu().numpy()
    save_path = Path(save_path)
    for idx, path in enumerate(paths):
        path = Path(path)
        cls = path.parent.name

        vision = cv2.cvtColor(cv2.resize(cv2.imread(path.as_posix()), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        abnormal = abnormals[idx][1]
        vision = apply_ad_scoremap(vision, mask)
        vision = cv2.cvtColor(vision, cv2.COLOR_RGB2BGR)  # BGR

        save_vision = save_path / "images" / cls_name[idx] / cls
        save_vision.mkdir(parents=True, exist_ok=True)
        
        filename = f"{path.stem}_{abnormal:.2f}{path.suffix}"

        save_file = save_vision / filename
        cv2.imwrite(save_file.as_posix(), vision)
