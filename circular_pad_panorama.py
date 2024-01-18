"""

Semi-scuffed circular padding implementation for panoramas

kernel_size - give kernel size which is used for padding
padding - either int or tuple of 4 values for padding
dilation - dilation of kernel for additional padding values

This will only work for 2d convolutions. Other convolutions fuck off. This is for panorama images specifically

NOTE - This will not quite work with padding modes "same" and "valid". You need to be as specific as possible

"""


from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# Circular padding for panoramas - includes basic circularity in width and inverted reflect in height
# This module will pad image, based on convolution parameters (so output size = input size, when stride=1)
# Not as flexible as Normal padding, since you can only differ height/width, while there you can do all 4 sides separately
class CircularPadPanorama(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        dilation: Union[int, tuple] = 1,
    ) -> None:
        super().__init__()

        # Asserts for functionality
        assert type(kernel_size) == int or (
            type(kernel_size) == tuple and len(kernel_size) == 2
        ), "Kernel size is either single int for square kernel or 2 dimensional tuple!"
        assert type(stride) == int or (
            type(stride) == tuple and len(stride) == 2
        ), "Stride is either single int for pad_both directions or 2 dimensional tuple!"
        assert type(dilation) == int or (
            type(dilation) == tuple and len(dilation) == 2
        ), "Dilation is either single int for square kernel or 2 dimensional tuple!"

        # Initial setup for padding values, based on given parameters upon initialization
        if type(kernel_size) == tuple:
            self.h_kernel, self.w_kernel = kernel_size
        else:
            self.h_kernel, self.w_kernel = kernel_size, kernel_size

        if type(stride) == tuple:
            self.h_stride, self.w_stride = stride
        else:
            self.h_stride, self.w_stride = stride, stride

        if type(dilation) == tuple:
            self.h_dilation, self.w_dilation = dilation
        else:
            self.h_dilation, self.w_dilation = dilation, dilation

        # Now calculate padding sizes for each side, so we can do the padding operations correctly
        # Easy calculation is - (kernel size - 1) / padding
        self.pad_left = int(((self.w_kernel - 1) * (self.w_dilation)) / 2)
        self.pad_right = int(((self.w_kernel - 1) * (self.w_dilation)) / 2)
        self.pad_top = int(((self.h_kernel - 1) * (self.h_dilation)) / 2)
        self.pad_bot = int(((self.h_kernel - 1) * (self.h_dilation)) / 2)

    def forward(self, x: torch.Tensor):
        # Get dimensions, just in case we need them
        B, C, H, W = x.size()

        # Padding happens in 3 steps:
        # 1 - We pad width with circularity
        # 2 - We pad height with reflect padding
        # 3 - We flip relfect padding, to represent correct circularity in height

        # 1st step - circular padding on width:
        x = F.pad(x, pad=(self.pad_left, self.pad_right, 0, 0), mode="circular")

        # 2nd step - reflect padding on height:
        x = F.pad(x, pad=(0, 0, self.pad_top, self.pad_bot), mode="reflect")

        # 3rd step - flip reflect padding on top and bottom parts
        x[:, :, : self.pad_top, :] = x[:, :, : self.pad_top, :].flip(3)
        x[:, :, -self.pad_bot :, :] = x[:, :, -self.pad_bot :, :].flip(3)

        return x


# Normal padding as it should be - only giving padding parameters
# This module will pad straight to given padding parameter
class CircularPadPanoramaNormal(nn.Module):
    def __init__(self, padding: Union[int, tuple]) -> None:
        super().__init__()
        """Circular padding module for panorama images. Since bottom part is black nadir, we don't want to circular pad in height,
        rather, we want to use reflect and invert padding for height, so it's correct to panorama image.
        
        params:
        padding - single integer or tuple of 2 or 4 integers. If single integer, will pad all image with said padding.
        If passed 2 int tuple, will pad image with (height, width). If passed 4 int tuple, will pad image with (left, right, top, bottom)
        """

        assert type(padding) == int or (
            type(padding) == tuple and (len(padding) == 2 or len(padding == 4))
        ), "Padding must be either single int or tuple of ints!"

        # self.pad_left = None
        # self.pad_right = None
        # self.pad_top = None
        # self.pad_bot = None

        if type(padding) == tuple:
            if len(padding) == 2:
                self.pad_top, self.pad_bot = padding[0], padding[0]
                self.pad_left, self.pad_right = padding[1], padding[1]
            else:
                self.pad_left, self.pad_right, self.pad_top, self.pad_bot = padding
        else:
            self.pad_left, self.pad_right, self.pad_top, self.pad_bot = (
                padding,
                padding,
                padding,
                padding,
            )

    def forward(self, x):
        # 1. - Pad circular on width:
        x = F.pad(x, (self.pad_left, self.pad_right, 0, 0), mode="circular")

        # 2. - Pad reflect on height:
        x = F.pad(x, (0, 0, self.pad_top, self.pad_bot), mode="reflect")

        # 3. - Invert top and bottom padding:
        x[:, :, : self.pad_top, :] = x[:, :, : self.pad_top, :].flip(3)
        x[:, :, -self.pad_bot :, :] = x[:, :, -self.pad_bot :, :].flip(3)

        return x


if __name__ == "__main__":
    pad = CircularPadPanoramaNormal(padding=(2, 2))

    pad_r = CircularPadPanorama(kernel_size=5, dilation=1)

    t = torch.arange(100, dtype=torch.float32).reshape(10, 10).unsqueeze(0).unsqueeze(0)

    t_pad = pad(t)
    t_pad_r = pad_r(t)

    conv = torch.nn.Conv2d(1, 10, kernel_size=3, stride=1)

    out = conv(t_pad)

    print(f"input size: {t.size()}")
    print(f"output size: {out.size()}")
    print("If original input size matches output size, good job!")

    pass
