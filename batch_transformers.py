"""

Batch transformation functions


"""


import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from PIL import Image
import collections

RANDOM_RESOLUTIONS = [512, 768, 1024, 1280, 1536]


class BatchRandomResolution(object):
    def __init__(self, size=None, interpolation=transforms.InterpolationMode.BILINEAR):
        assert (
            isinstance(size, int)
            # or (isinstance(size, collections.Iterable) and len(size) == 2)
            or (size is None)
            or (isinstance(size, tuple) and len(size) == 2)
        )
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        if self.size is None:
            h, w = imgs[0].size
            max_idx = 0
            for i in range(len(RANDOM_RESOLUTIONS)):
                if h > RANDOM_RESOLUTIONS[i] and w > RANDOM_RESOLUTIONS[i]:
                    max_idx += 1
            idx = np.random.randint(max_idx)
            self.size = RANDOM_RESOLUTIONS[idx]
        return [transforms.Resize(self.size, self.interpolation)(img) for img in imgs]


class BatchTestResolution(object):
    def __init__(self, size=None, interpolation=transforms.InterpolationMode.BILINEAR):
        assert (
            isinstance(size, int)
            or (isinstance(size, tuple) and len(size) == 2)
            or (size is None)
        )
        self.size = size[0] if type(size) is tuple else size
        self.size_true = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        h, w = imgs[0].size
        if h > self.size and w > self.size:
            return [
                transforms.Resize(self.size_true, self.interpolation)(img)
                for img in imgs
            ]
        else:
            return imgs


class BatchToTensor(object):
    def __call__(self, imgs):
        return [transforms.ToTensor()(img) for img in imgs]


class BatchRGBToGray(object):
    def __call__(self, imgs):
        return [
            img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2:, :, :] * 0.114
            for img in imgs
        ]


class BatchRGBToYCbCr(object):
    def __call__(self, imgs):
        return [
            torch.stack(
                (
                    0.0 / 256.0
                    + img[0, :, :] * 0.299000
                    + img[1, :, :] * 0.587000
                    + img[2, :, :] * 0.114000,
                    128.0 / 256.0
                    - img[0, :, :] * 0.168736
                    - img[1, :, :] * 0.331264
                    + img[2, :, :] * 0.500000,
                    128.0 / 256.0
                    + img[0, :, :] * 0.500000
                    - img[1, :, :] * 0.418688
                    - img[2, :, :] * 0.081312,
                ),
                dim=0,
            )
            for img in imgs
        ]


class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack(
            (
                img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.0) * 1.402,
                img[:, 0, :, :]
                - (img[:, 1, :, :] - 128 / 256.0) * 0.344136
                - (img[:, 2, :, :] - 128 / 256.0) * 0.714136,
                img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.0) * 1.772,
            ),
            dim=1,
        )


class RGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack(
            (
                0.0 / 256.0
                + img[:, 0, :, :] * 0.299000
                + img[:, 1, :, :] * 0.587000
                + img[:, 2, :, :] * 0.114000,
                128.0 / 256.0
                - img[:, 0, :, :] * 0.168736
                - img[:, 1, :, :] * 0.331264
                + img[:, 2, :, :] * 0.500000,
                128.0 / 256.0
                + img[:, 0, :, :] * 0.500000
                - img[:, 1, :, :] * 0.418688
                - img[:, 2, :, :] * 0.081312,
            ),
            dim=1,
        )
