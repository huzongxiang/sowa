from typing import Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torch.utils.data as data


@dataclass
class DatasetItem:
    img: Image.Image
    img_mask: Any
    cls_name: str
    anomaly: int
    img_path: Path


class BaseDataset(data.Dataset, ABC):
    """
    An abstract base dataset class for image datasets that provides common functionalities
    and supports both preloading and on-demand loading.

    Args:
        root (Path or str): Root directory path of the dataset.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        mask_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        preload (bool): Specifies whether to preload data into memory or not.
    """
    def __init__(self, root, transform=None, mask_transform=None, preload=False, meta_file="meta.json"):
        self.root = Path(root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.preload = preload
        self.meta_file = meta_file

        self.data = []
        self.preloaded_data = []
        self._load_meta()

        if self.preload:
            self._preload_data()

    def _load_meta(self, mode='test'):
        """Load metadata from JSON file and initialize dataset items."""
        meta_path = self.root / self.meta_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

        with meta_path.open('r') as f:
            meta_info = json.load(f)[mode]
            self.data = [(cls_info['img_path'], cls_info['mask_path'], cls_info['cls_name'], cls_info['anomaly'])
                             for cls_list in meta_info.values() for cls_info in cls_list]

    def _preload_data(self):
        """Preload all images and masks into memory."""
        for index, (img_path, mask_path, cls_name, anomaly) in enumerate(self.data):
            img = self._load_image(self.root / img_path)
            mask = self._load_mask(self.root / mask_path, (img.height, img.width))

            assert img.size == mask.size, f"Size mismatch: image {img.size} vs mask {mask.size} classname: {cls_name} anomaly: {anomaly} index: {index}"

            if self.transform is not None:
                img = self.transform(img)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

            self.preloaded_data.append((img, mask, cls_name, anomaly))

    def __len__(self):
        return len(self.preloaded_data) if self.preload else len(self.data)

    @abstractmethod
    def __getitem__(self, index):
        """Retrieve an item by its index, either from preloaded data or by on-demand loading."""
        if self.preload:
            return self.preloaded_data[index]
        else:
            return self._get_on_demand(index)

    @abstractmethod
    def _get_on_demand(self, index):
        """Get data by index on demand. Subclasses must implement this method."""
        pass

    def _load_image(self, path):
        """Load an image file."""
        return Image.open(path).convert('RGB')

    def _load_mask(self, path, default_size):
        """Load a mask file or create an empty mask if not found."""
        if path.exists() and (path.stat().st_size > 0) and path.is_file():
            mask = np.array(Image.open(path).convert('L'))
            return Image.fromarray((mask > 0).astype(np.uint8) * 255, 'L')
        else:
            return Image.fromarray(np.zeros(default_size, dtype=np.uint8), 'L')


class VisaDataset(BaseDataset):
    def __getitem__(self, index):
        """Retrieve an item by its index, either from preloaded data or by on-demand loading."""
        if self.preload:
            img, mask, cls_name, anomaly = self.preloaded_data[index]
        else:
            img, mask, cls_name, anomaly = self._get_on_demand(index)

        img_path = self.root / self.data[index][0]

        return {
            "image": img, 
            "image_mask": mask, 
            "cls_name": cls_name, 
            "anomaly": anomaly,
            "image_path": img_path.as_posix(),
        }

    def _get_on_demand(self, index):
        """Retrieve an item by its index for on-demand loading."""
        img_path, mask_path, cls_name, anomaly = self.data[index]
        img = self._load_image(self.root / img_path)
        mask = self._load_mask(self.root / mask_path, (img.height, img.width))

        assert img.size == mask.size, f"Size mismatch: image {img.size} vs mask {mask.size} classname: {cls_name} anomaly: {anomaly} index: {index}"

        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask, cls_name, anomaly


class MVTecDataset(BaseDataset):
    def __init__(self, root, transform=None, mask_transform=None, preload=False, meta_file="meta.json", aug_rate=0.0):
        self.aug_rate = aug_rate
        super().__init__(root, transform, mask_transform, preload, meta_file)

    def _preload_data(self):
        """Preload all images and masks into memory."""
        for index, (img_path, mask_path, cls_name, anomaly) in enumerate(self.data):
            # Optionally apply image combining as an augmentation technique
            if random.random() < self.aug_rate:
                img, mask = self._combine_img(cls_name)
            else:
                img = self._load_image(self.root / img_path)
                mask = self._load_mask(self.root / mask_path, (img.height, img.width))

            assert img.size == mask.size, f"Size mismatch: image {img.size} vs mask {mask.size} classname: {cls_name} anomaly: {anomaly} index: {index}"

            if self.transform is not None:
                img = self.transform(img)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

            self.preloaded_data.append((img, mask, cls_name, anomaly))

    def __getitem__(self, index):
        """Retrieve an item by its index, either from preloaded data or by on-demand loading."""
        if self.preload:
            img, mask, cls_name, anomaly = self.preloaded_data[index]
        else:
            img, mask, cls_name, anomaly = self._get_on_demand(index)

        img_path = self.root / self.data[index][0]

        return {
            "image": img, 
            "image_mask": mask, 
            "cls_name": cls_name, 
            "anomaly": anomaly,
            "image_path": img_path.as_posix(),
        }

    def _get_on_demand(self, index):
        """Retrieve an item by its index for on-demand loading."""
        img_path, mask_path, cls_name, anomaly = self.data[index]
        # Optionally apply image combining as an augmentation technique
        if random.random() < self.aug_rate:
            img, mask = self._combine_img(cls_name)
        else:
            img = self._load_image(self.root / img_path)
            mask = self._load_mask(self.root / mask_path, (img.width, img.height))

        assert img.size == mask.size, f"Size mismatch: image {img.size} vs mask {mask.size} classname: {cls_name} anomaly: {anomaly} index: {index}"

        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask, cls_name, anomaly

    def _combine_img(self, cls_name):
        """Combine images for augmentation."""
        cls_path = self.root / cls_name / 'test'
        img_ls = []
        mask_ls = []

        for i in range(4):
            defect = list(cls_path.iterdir())
            random_defect = random.choice(defect)
            files = list(random_defect.iterdir())
            random_file = random.choice(files)
            img = self._load_image(random_file)

            if random_defect.name == 'good':
                img_mask = Image.fromarray(np.zeros((img.width, img.height), dtype=np.uint8), 'L')
            else:
                mask_path = self.root / cls_name / 'ground_truth' / random_defect.name / (random_file.stem + '_mask.png')
                img_mask = self._load_mask(mask_path, (img.width, img.height))

            img_ls.append(img)
            mask_ls.append(img_mask)

        # Combine images into one larger image
        image_width, image_height = img_ls[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        result_mask = Image.new("L", (2 * image_width, 2 * image_height))

        for i, (img, mask) in enumerate(zip(img_ls, mask_ls)):
            x = (i % 2) * image_width
            y = (i // 2) * image_height
            result_image.paste(img, (x, y))
            result_mask.paste(mask, (x, y))

        return result_image, result_mask
