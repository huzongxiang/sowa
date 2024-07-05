from abc import ABC, abstractmethod
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.utils.data as data


class KShotDataset(data.Dataset, ABC):
    def __init__(self, root, k_shot=1, transform=None, mask_transform=None, preload=True, meta_file="meta.json"):
        self.root = Path(root)
        self.k_shot = k_shot
        self.transform = transform
        self.mask_transform = mask_transform
        self.preload = preload
        self.data = []
        self.preloaded_data = []
        self.meta_file = meta_file

        self._load_meta()
        if self.preload:
            self._preload_data()

    def _load_meta(self, mode='train'):
        meta_path = self.root / self.meta_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

        with meta_path.open('r') as f:
            meta_info = json.load(f)[mode]

        for cls_name, items in meta_info.items():
            if len(items) >= self.k_shot:
                sampled_items = random.sample(items, self.k_shot)
            else:
                sampled_items = items

            self.data.append((cls_name, sampled_items))

    def _preload_data(self):
        for cls_name, items in self.data:
            preloaded_items = []
            for item in items:
                img_path = self.root / item['img_path']
                mask_path = self.root / item['mask_path']
                img = self._load_image(img_path)
                mask = self._load_mask(mask_path, (img.height, img.width))

                preloaded_items.append((img, mask))
            self.preloaded_data.append((cls_name, preloaded_items))

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def _load_mask(self, path, default_size):
        if path.exists() and (path.stat().st_size > 0) and path.is_file():
            mask = np.array(Image.open(path).convert('L'))
            return Image.fromarray((mask > 0).astype(np.uint8) * 255, 'L')
        else:
            return Image.fromarray(np.zeros(default_size, dtype=np.uint8), 'L')


class VisaKShotDataset(KShotDataset):
    def __getitem__(self, index):
        """Retrieve an item by its index."""
        cls_name, items = self.data[index]

        if self.preload:
            imgs, masks = zip(*self.preloaded_data[index][1])
        else:
            imgs, masks = [], []
            for item in items:
                img_path = self.root / item['img_path']
                mask_path = self.root / item['mask_path']

                img = self._load_image(img_path)
                mask = self._load_mask(mask_path, (img.height, img.width)) if 'mask_path' in item else Image.fromarray(np.zeros(img.size, dtype=np.uint8), 'L')

                imgs.append(img)
                masks.append(mask)

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        else:
            raise ValueError("Transform must be provided and not None.")
        
        if self.mask_transform is not None:
            masks = [self.mask_transform(mask) for mask in masks]
        else:
            raise ValueError("Mask transform must be provided and not None.")
        
        return {'image': torch.stack(imgs), 'mask': torch.stack(masks), 'cls_name': cls_name}


class MVTecKShotDataset(KShotDataset):
    def __getitem__(self, index):
        """Retrieve an item by its index."""
        cls_name, items = self.data[index]

        if self.preload:
            imgs, masks = zip(*self.preloaded_data[index][1])
        else:
            imgs, masks = [], []
            for item in items:
                img_path = self.root / item['img_path']
                mask_path = self.root / item['mask_path']

                img = self._load_image(img_path)
                mask = self._load_mask(mask_path, (img.height, img.width))

                imgs.append(img)
                masks.append(mask)

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        else:
            raise ValueError("Transform must be provided and not None.")
        
        if self.mask_transform is not None:
            masks = [self.mask_transform(mask) for mask in masks]
        else:
            raise ValueError("Mask transform must be provided and not None.")

        return {'image': torch.stack(imgs), 'mask': torch.stack(masks), 'cls_name': cls_name}
