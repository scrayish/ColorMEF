"""

MEF SSIM metric implementation
Most functions from original files actually hold up. Shows, how well PyTorch has aged
loss_scaling(y) = sqrt(1-x^2), kur x = y/2500 , un y = [-2500,2500]  # Adjusted to use actual image sizes
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / torch.sum(gauss)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, window_size / 6.0).unsqueeze(dim=1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(dim=0).unsqueeze(dim=0)
    window = (
        torch.Tensor(
            _2D_window.expand(1, channel, window_size, window_size).contiguous()
        )
        / channel
    )
    return window


def pad_conv2d(input, window, padding, pad_fn: nn.Module):
    input = pad_fn(padding)(input)
    return F.conv2d(input, window)


def _mef_ssim(
    X,
    Ys,
    window,
    ws,
    denom_g,
    denom_l,
    C1,
    C2,
    is_lum=False,
    full=False,
    scale=False,
    full_structure=False,
    pad_fn: nn.Module = nn.ZeroPad2d,
):
    K, C, H, W = Ys.size()

    # compute statistics of the reference latent image Y
    muY_seq = pad_conv2d(Ys, window, padding=ws // 2, pad_fn=pad_fn).view(K, H, W)
    # muY_seq = F.conv2d(Ys, window, padding=ws // 2).view(K, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = (
        pad_conv2d(Ys * Ys, window, padding=ws // 2, pad_fn=pad_fn).view(K, H, W)
        - muY_sq_seq
    )
    # sigmaY_sq_seq = (
    #     F.conv2d(Ys * Ys, window, padding=ws // 2).view(K, H, W) - muY_sq_seq
    # )
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = pad_conv2d(X, window, padding=ws // 2, pad_fn=pad_fn).view(H, W)
    # muX = F.conv2d(X, window, padding=ws // 2).view(H, W)
    muX_sq = muX * muX
    sigmaX_sq = (
        pad_conv2d(X * X, window, padding=ws // 2, pad_fn=pad_fn).view(H, W) - muX_sq
    )
    # sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2).view(H, W) - muX_sq

    # compute correlation term
    sigmaXY = (
        pad_conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, pad_fn=pad_fn).view(
            K, H, W
        )
        - muX.expand_as(muY_seq) * muY_seq
    )
    # sigmaXY = (
    #     F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2).view(K, H, W)
    #     - muX.expand_as(muY_seq) * muY_seq
    # )

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(H, W)

    if full_structure:
        neg_sigmaY_sq, patch_index_min = torch.min(sigmaY_sq_seq, dim=0)
        patch_index_mid = torch.where(
            patch_index == patch_index_min,
            patch_index,
            3 - (patch_index + patch_index_min),
        )
        cs_map_min = torch.gather(
            cs_seq.view(K, -1), 0, patch_index_min.view(1, -1)
        ).view(H, W)
        cs_map_mid = torch.gather(
            cs_seq.view(K, -1), 0, patch_index_mid.view(1, -1)
        ).view(H, W)
        # Quality map from all possible maps, with weighted coefficients (currently hard-coded, could be researched as well)
        cs_map = cs_map * 0.9 + cs_map_mid * 0.09 + cs_map_min * 0.01

    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(-((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    # Here, both, potentially
    if scale:
        scale_map = (torch.linspace(-H // 2, H // 2, H) / (H // 2)) ** 2
        scale_map = scale_map.unsqueeze(1).repeat(1, W).to(l_map.device)
        # Scale luma only when it's not constant
        if is_lum:
            l_map = l_map * scale_map
        cs_map = cs_map * scale_map

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q


def mef_ssim(X, Ys, window_size=11, is_lum=False, scale=False):
    (_, channel, _, _) = Ys.size()
    window = create_window(window_size, channel)

    if Ys.is_cuda:
        window = window.cuda(Ys.get_device())
    window = window.type_as(Ys)

    return _mef_ssim(
        X, Ys, window, window_size, 0.08, 0.08, 0.01**2, 0.03**2, is_lum, scale
    )


def mef_msssim(
    X,
    Ys,
    window,
    ws,
    denom_g,
    denom_l,
    C1,
    C2,
    is_lum=False,
    scale=False,
    full_structure=False,
    pad_fn: nn.Module = nn.ZeroPad2d,
):
    # beta = torch.Tensor([0.0710, 0.4530, 0.4760])
    # beta = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    # beta = torch.Tensor([1, 1, 1, 1, 1])
    beta = torch.Tensor([1])
    if Ys.is_cuda:
        window = window.cuda(Ys.get_device())
        beta = beta.cuda(Ys.get_device())

    window = window.type_as(Ys)

    levels = beta.size()[0]
    l_i = []
    cs_i = []
    for _ in range(levels):
        l, cs = _mef_ssim(
            X,
            Ys,
            window,
            ws,
            denom_g,
            denom_l,
            C1,
            C2,
            is_lum=is_lum,
            full=True,
            scale=scale,
            full_structure=full_structure,
            pad_fn=pad_fn,
        )
        l_i.append(l)
        cs_i.append(cs)

        X = F.avg_pool2d(X, (2, 2))
        Ys = F.avg_pool2d(Ys, (2, 2))

    Ql = torch.stack(l_i)
    Qcs = torch.stack(cs_i)

    return (Ql[levels - 1] ** beta[levels - 1]) * torch.prod(Qcs**beta)


# MEF SSIM definition
class MEFSSIM(nn.Module):
    def __init__(
        self,
        window_size=11,
        channel=3,
        sigma_g=0.2,
        sigma_l=0.2,
        c1=0.01,
        c2=0.03,
        is_lum=False,
        scale=False,
    ):
        super(MEFSSIM, self).__init__()

        # Regular parameters
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(self.window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.c1 = c1**2
        self.c2 = c2**2
        self.is_lum = is_lum
        self.scale = scale

    def forward(self, X, Ys):
        channel = Ys.size()[1]

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, self.channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(
            X,
            Ys,
            window,
            self.window_size,
            self.denom_g,
            self.denom_l,
            self.C1,
            self.C2,
            self.is_lum,
            self.scale,
        )


class MEF_MSSSIM(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        channel=3,
        sigma_g=0.2,
        sigma_l=0.2,
        c1=0.01,
        c2=0.03,
        is_lum=False,
        scale=False,
        full_structure=False,
        pad_fn: nn.Module = nn.ZeroPad2d,
    ):
        super(MEF_MSSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum
        self.scale = scale
        self.full_structure = full_structure
        self.pad_fn = pad_fn

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return mef_msssim(
            X,
            Ys,
            window,
            self.window_size,
            self.denom_g,
            self.denom_l,
            self.C1,
            self.C2,
            self.is_lum,
            self.scale,
            self.full_structure,
            self.pad_fn,
        )
