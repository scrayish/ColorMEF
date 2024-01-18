"""

Image dataset file for formatting data


"""


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


class ImageDataset(Dataset):
    def __init__(
        self,
        data_path,
        hr_transform,
        lr_transform,
        fuse_expos,
        remove_dark=False,
        inference=False,
    ):
        # Making some assumptions here
        self.data_path = data_path
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.remove_dark = remove_dark
        self.fuse_expos = fuse_expos
        self.inference = inference
        assert (
            len(self.fuse_expos) > 1
        ), "Need at least 2 exposures to fuse, check input arguments"

        # Save lists of specified fused expos, paths must exist beforehand. I am not checking if they exist, it can simply die
        self.data_lists = {
            f"{expo}": sorted(list(Path(self.data_path + f"/{expo}").iterdir()))
            for expo in self.fuse_expos
        }

    def __len__(self):
        return len(
            self.data_lists[list(self.data_lists.keys())[0]]
        )  # Take first key, no matter which one it is

    def __getitem__(self, item):
        # Load images from lists and format them as needed for model to train
        img_seq = [
            Image.open(str(self.data_lists[t_exp][item])) for t_exp in self.fuse_expos
        ]
        I_hr = self.hr_transform(img_seq)
        I_lr = self.lr_transform(img_seq)

        I_hr = torch.stack(I_hr, 0).contiguous()
        I_lr = torch.stack(I_lr, 0).contiguous()

        if self.inference:
            image_name = self.data_lists[self.fuse_expos[0]][item].name
            return image_name, {"I_hr": I_hr, "I_lr": I_lr}
        else:
            return {"I_hr": I_hr, "I_lr": I_lr}
