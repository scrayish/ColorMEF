"""

Guided filter implementation in PyTorch


"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolutional self-attention module
class ConvSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.mid_channels = 1 if channels // 4 == 0 else channels // 4

        self.scale = 8

        self.pool_in = nn.AvgPool2d(self.scale, self.scale)

        self.conv_q = nn.Conv2d(
            in_channels=channels, out_channels=self.mid_channels, kernel_size=1
        )
        self.conv_k = nn.Conv2d(
            in_channels=channels, out_channels=self.mid_channels, kernel_size=1
        )
        self.conv_v = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=1
        )

        self.conv_out = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1)

        # Initialize learnable scale parameter to a random value
        self.gamma = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x_in = self.pool_in(x)

        # Shapes of network
        B, C, H, W = x_in.size()

        # Get feature maps
        q = self.conv_q.forward(x_in)
        k = self.conv_k.forward(x_in)
        v = self.conv_v.forward(x_in)

        # Do the attention stuff
        q = q.flatten(2).transpose(1, 2)  # B, HW, C
        k = k.flatten(2)  # B, C, HW
        v = v.flatten(2)  # B, C, HW

        self_attn = torch.softmax(torch.matmul(q, k), 0)  # B, HW, HW

        out = torch.matmul(v, self_attn) * self.gamma  # B, C, HW

        # Reshape, sum and return
        out = F.interpolate(
            out.view(B, C, H, W),
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=True,
        )

        # Flatten channels to one
        return self.conv_out(out + x)


# Attentional Kernel Learning Module
class AKL(nn.Module):
    def __init__(self, channels, kernel_out_size: int, pad: int, pad_fn: nn.Module):
        super().__init__()

        # Guide is the image and target is the mask (in paper they give example as high-res rgb image)
        # But, high res image would never work here, because you have to scale image down massively
        # Probably fucks the entire network config in the ass.
        self.guide_kernel = nn.Sequential(
            self.conv_stack(channels, channels, 3, 1, pad, pad_fn),
            self.conv_stack(channels, kernel_out_size**2, 3, 1, pad, pad_fn),
        )
        self.target_kernel = nn.Sequential(
            self.conv_stack(channels, channels, 3, 1, pad, pad_fn),
            self.conv_stack(channels, kernel_out_size**2, 3, 1, pad, pad_fn),
        )

        # Module for processing the input of both thingies - going to use attention mechanism
        # instead of U-Net, so we have global receptive field
        self.attention = ConvSelfAttention(channels * 2)

    def conv_stack(self, c_in, c_out, kernel, stride, pad, pad_fn):
        return nn.Sequential(
            pad_fn(pad),
            nn.Conv2d(c_in, c_out, kernel, stride),
            nn.GroupNorm(1, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, guide, target):
        # Concatenate features for the middle branch of the model
        features = torch.cat([guide, target], dim=1)
        attention = self.attention(features)

        # Individual feature branches
        target = torch.softmax(self.target_kernel(target), 1)
        guide = torch.softmax(self.guide_kernel(guide), 1)

        # Attention mathematics
        fuse_kernel = target * (1 - attention) + guide * attention

        # Additional math, which is not documented in the paper (very nice)
        # This seems like kernels get normalized in some way
        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0

        # Final kernel definition
        fuse_kernel = fuse_kernel / abs_kernel_sum
        fuse_kernel = F.interpolate(
            fuse_kernel, scale_factor=2, mode="bilinear", align_corners=True
        )

        return fuse_kernel


# Wrap AKL with initial layers and other stuff (hopefully it helps)
class AttentionalFilter(nn.Module):
    def __init__(self, channels, features, pad: int, pad_fn: nn.Module) -> None:
        super().__init__()

        self.init_guide = nn.Sequential(
            pad_fn(pad),
            nn.Conv2d(channels, features, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            pad_fn(pad),
            nn.Conv2d(features, features, kernel_size=3),
        )
        self.init_target = nn.Sequential(
            pad_fn(pad),
            nn.Conv2d(channels, features, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            pad_fn(pad),
            nn.Conv2d(features, features, kernel_size=3),
        )
        self.akl = AKL(channels=features, kernel_out_size=1, pad=pad, pad_fn=pad_fn)

    def forward(self, guide, target):
        guide = self.init_guide(guide)
        target = self.init_target(target)
        out = self.akl(guide, target)
        return out


# box filter definitions
def diff_x(data_in: torch.Tensor, r):
    assert data_in.dim() == 4

    left = data_in[:, :, r : 2 * r + 1]
    middle = data_in[:, :, 2 * r + 1 :] - data_in[:, :, : -2 * r - 1]
    right = data_in[:, :, -1:] - data_in[:, :, -2 * r - 1 : -r - 1]

    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(data_in: torch.Tensor, r):
    assert data_in.dim() == 4

    left = data_in[:, :, :, r : 2 * r + 1]
    middle = data_in[:, :, :, 2 * r + 1 :] - data_in[:, :, :, : -2 * r - 1]
    right = data_in[:, :, :, -1:] - data_in[:, :, :, -2 * r - 1 : -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


# Guided filter definitions
class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(self.r)

    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # Are these assertions really necessary?
        # assert n_lrx == n_lry and n_lry == n_hrx
        # assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        # assert h_lrx == h_lry and w_lrx == w_lry
        # assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        # Wrap with nn.Parameter to get gradient calculating for N so backward works
        # as Variable doesn't enable any gradient anymore, it seems
        N = self.boxfilter(
            nn.Parameter(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        )

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=False)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=False)

        return mean_A * hr_x + mean_b


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(self.r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(
            nn.Parameter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))
        )

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b


class ConvGuidedFilter(nn.Module):
    def __init__(self, pad_fn, radius=1, norm=nn.GroupNorm):
        super(ConvGuidedFilter, self).__init__()

        # Introduce padding function here as well, since it is necessary to leave this regular
        self.box_filter = torch.nn.Sequential(
            pad_fn(radius), nn.Conv2d(1, 1, kernel_size=3, dilation=radius, bias=False)
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, bias=False),
            norm(2, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(2, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
        )
        self.box_filter[1].weight.data[...] = 1.0  # EZ fix lol

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N
        ## mean_y
        mean_y = self.box_filter(y_lr) / N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * x_hr + mean_b


# Scuffed guided filter is based on https://arxiv.org/pdf/2112.06401.pdf
# It doesn't fully work that way - since I use convolutional attention, instead of U-like network
# Thought process was that using this sort of contraption would reduce memory overhead
# I also still use regular parameters simliar to rest of filters. I could try cut out
# some of them (cov_xy and var_x) and use simply inputs. But I'll test this setup first
class ScuffedGuidedFilter(nn.Module):
    def __init__(self, pad: int, pad_fn: nn.Module) -> None:
        super().__init__()

        self.filter = AttentionalFilter(1, 32, pad, pad_fn)
        self.box_filter = torch.nn.Sequential(
            pad_fn(pad), nn.Conv2d(1, 1, kernel_size=3, dilation=pad, bias=False)
        )

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N

        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        # Get A straight from our stuff
        A = self.filter(var_x, cov_xy)
        # b
        b = mean_y - A * mean_x

        # mean_A, mean_B
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * x_hr + mean_b
