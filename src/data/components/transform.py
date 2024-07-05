import torch
import torchvision.transforms as transforms


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class ImageTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, images):
        return transforms.Compose(
            [
                transforms.Resize(size=(self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(size=(self.image_size, self.image_size)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )(images)


class MaskTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, images):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )(images)