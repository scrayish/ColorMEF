"""

Transformer v5 architecture - the hard one. In reality it shouldn't be too hard, but, you never know
TODO: Re-write this model, so it is actually readable instad of being totally trash
11.05.2023. Alright, I rewrote parts of the model. Have yet to test if it actually works (it should).
Also, not much that I could reduce from this point.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from guided_filter import (
    GuidedFilter,
    FastGuidedFilter,
    ConvGuidedFilter,
    ScuffedGuidedFilter,
)

from circular_pad_panorama import CircularPadPanoramaNormal
from einops import rearrange
from einops.layers.torch import Rearrange


# Override regular identity with custom one
class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return x, y


# Interpolation wrapper (interpolates to scale)
class Interpolate(nn.Module):
    def __init__(
        self, scale_factor, mode: str = "bilinear", align_corners: bool = True
    ) -> None:
        super(Interpolate, self).__init__()

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


# Feedforward layer for transformer unit
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


# Dictionary Convolutional Unit (Uses CSC inside itself for some memes)
# I actually don't know wether it actually uses them. I simply assume it does, because
# the schematics show it like this. I used paper scheme to implement this. Their code seemingly does
# very similar thing (except they have swapped around subtraction, which can cause different results)
class DCU(nn.Module):
    def __init__(
        self,
        img_channels,
        code_channels,
        kernel_size,
        padding: int,
        pad_fn: nn.Module,
    ):
        super(DCU, self).__init__()

        # Define all the stuff I need
        self.decoder = nn.Sequential(
            pad_fn(padding), nn.Conv2d(code_channels, img_channels, kernel_size)
        )
        self.encoder = nn.Sequential(
            pad_fn(padding), nn.Conv2d(img_channels, code_channels, kernel_size)
        )
        self.norm = nn.GroupNorm(
            num_groups=code_channels // 4, num_channels=code_channels
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, data, code):
        # Do the meme
        out = self.decoder(code) - data
        out = self.encoder(out) + code
        out = self.activation(self.norm(out))
        return out


# Custom multi-head attention unit, so we can do some other memes with it
class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        readout: str,
        relative_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.readout = readout
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.relative_embedding = relative_embedding

    # Relative positional embedding for values inside attention - stolen from https://arxiv.org/pdf/2112.01526.pdf
    def relative_pos_embedding(self, k):
        # Handle class embedding, if it exists, set index
        class_idx = 1 if self.readout else 0

        # Set distances
        dist_h = torch.arange()

        pass

    def forward(self, q, k, v):
        # Self attention implementation and stuff
        B, N, C = q.size()

        # (Batch, N-heads, Sequence, Features // N-heads) is resulting dimensions
        q = q.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        pass


# Transformer unit (single layer)
class TransformerUnit(nn.Module):
    def __init__(self, dim, heads, mlp_dim) -> None:
        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads!"

        self.dim = dim
        self.c_attention = torch.nn.Linear(dim, 3 * dim)
        self.transformer = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(dim)
        self.norm_2 = torch.nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mlp_dim=mlp_dim)

    def forward(self, x):
        norm_1_out = self.norm_1.forward(x)
        q, k, v = self.c_attention.forward(norm_1_out).split(self.dim, dim=2)
        out_transformer = self.transformer.forward(q, k, v)[0] + x
        out = self.norm_2.forward(out_transformer)
        out = self.ff.forward(out) + out
        return out


# "Parallel" transformer unit, based on some sort of optimization
class ParallelTransformerUnit(nn.Module):
    def __init__(self, dim, heads, mlp_dim, use_layerscale) -> None:
        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads!"

        self.dim = dim
        self.c_attention = torch.nn.Linear(dim, 3 * dim)
        self.transformer_1 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.transformer_2 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(dim)
        self.norm_2 = torch.nn.LayerNorm(dim)
        self.ff_1 = FeedForward(dim, mlp_dim)
        self.ff_2 = FeedForward(dim, mlp_dim)
        # Layerscale scuffed implementation - https://arxiv.org/pdf/2103.17239v2.pdf
        if use_layerscale:
            self.gamma_1 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_3 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_4 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = self.gamma_2 = self.gamma_3 = self.gamma_4 = 1.0

    def forward(self, x, luma=None):
        assert (
            luma is None
        ), "Regular parallel unit doesn't take luma. Check your model!"

        x = self.norm_1.forward(x)
        q, k, v = self.c_attention.forward(x).split(self.dim, dim=2)
        x = (
            x
            + self.transformer_1(q, k, v)[0] * self.gamma_1
            + self.transformer_2(q, k, v)[0] * self.gamma_2
        )
        x = self.norm_2.forward(x)
        x = x + self.ff_1(x) * self.gamma_3 + self.ff_2(x) * self.gamma_4
        return x


# Mixed attention parallel block
class MixedAttentionParallelTransformerUnit(nn.Module):
    def __init__(self, dim, heads, mlp_dim, use_layerscale) -> None:
        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads!"

        self.dim = dim
        self.c_attention = torch.nn.Linear(dim, 3 * dim)
        self.m_attention = torch.nn.Linear(dim, 2 * dim)
        self.transformer_1 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.transformer_2 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(dim)
        self.norm_2 = torch.nn.LayerNorm(dim)
        self.ff_1 = FeedForward(dim, mlp_dim)
        self.ff_2 = FeedForward(dim, mlp_dim)
        # Layerscale scuffed implementation - https://arxiv.org/pdf/2103.17239v2.pdf
        if use_layerscale:
            self.gamma_1 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_3 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_4 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = self.gamma_2 = self.gamma_3 = self.gamma_4 = 1.0

    # Main juice is in here - because I need to pass stuff correctly
    def forward(self, x, luma=None):
        # assert that luma can't be none
        assert (
            luma is not None
        ), "Mixed attention must have input for luma. Check your model!"

        # Process input to get the vectors and stuff
        x = self.norm_1.forward(x)
        # This is for the first branch, where we do self-attention
        q, k, v = self.c_attention.forward(x).split(self.dim, dim=2)
        # This is for the second branch, where we do cross-attention between chroma and luma output
        k_luma, v_luma = self.m_attention.forward(luma).split(self.dim, dim=2)
        # Put values through the transformers
        x = (
            x
            + self.transformer_1.forward(q, k, v)[0] * self.gamma_1
            + self.transformer_2.forward(q, k_luma, v_luma)[0] * self.gamma_2
        )
        x = self.norm_2.forward(x)
        x = self.ff_1(x) * self.gamma_3 + self.ff_2(x) * self.gamma_4
        return x


# Parallel transformer unit with self attention and cross attention
class FullParallelTransformerUnit(nn.Module):
    def __init__(self, dim, heads, mlp_dim, use_layerscale) -> None:
        super().__init__()

        # Setup everything for getting value vectors out of inputs
        self.dim = dim
        self.luma_attention = nn.Linear(dim, 3 * dim)
        self.chroma_attention = nn.Linear(dim, 3 * dim)

        self.transformer_1 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.transformer_2 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.transformer_3 = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(dim)
        self.norm_2 = torch.nn.LayerNorm(dim)
        self.norm_3 = torch.nn.LayerNorm(dim)
        self.norm_4 = torch.nn.LayerNorm(dim)
        self.norm_5 = torch.nn.LayerNorm(dim)
        self.ff_1 = FeedForward(dim, mlp_dim)
        self.ff_2 = FeedForward(dim, mlp_dim)
        self.ff_3 = FeedForward(dim, mlp_dim)
        # Layerscale scuffed implementation - https://arxiv.org/pdf/2103.17239v2.pdf
        if use_layerscale:
            self.gamma_1 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_3 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_4 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = self.gamma_2 = self.gamma_3 = self.gamma_4 = 1.0

    def forward(self, luma, chroma):
        # Extend luma and chroma for transformers
        luma = self.norm_1(luma)
        chroma = self.norm_2(chroma)

        q_luma, k_luma, v_luma = self.luma_attention.forward(luma).split(
            self.dim, dim=-1
        )
        q_chroma, k_chroma, v_chroma = self.chroma_attention.forward(chroma).split(
            self.dim, dim=-1
        )

        # Put everything through transformer and set values
        luma = luma + self.transformer_1.forward(q_luma, k_luma, v_luma)[0]
        chroma = chroma + self.transformer_2.forward(q_chroma, k_chroma, v_chroma)[0]
        cross = self.transformer_3.forward(q_chroma, k_luma, v_luma)[0]

        # Sum together each individual component with cross attention output and normalize
        # Maybe here we don't do connections between cross and individual values, only do it in the end?
        luma = self.norm_3(luma + cross)
        chroma = self.norm_4(chroma + cross)
        cross = self.norm_5(cross)

        # Put everything through their respective feed forwards and sum what needs to be summed
        luma = luma + self.ff_1.forward(luma)
        chroma = chroma + self.ff_2.forward(chroma)  # Oops :D
        cross = cross + self.ff_3.forward(cross)

        # Now sum cross to luma and chroma - maybe we could try multiplication here?
        luma = luma + cross
        chroma = chroma + cross

        # Return luma and chroma
        return luma, chroma


# Process fusion embedding to get extra information on image to use for reconstruction
class AddFusionEmbedding(nn.Module):
    def __init__(self, start_index=1) -> None:
        super().__init__()

        self.start_index = start_index

    def forward(self, x: torch.Tensor):
        readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


# Fusion embedding, but have to handle somehow the dimensions
class CatFusionEmbedding(nn.Module):
    def __init__(self, dim, start_index=1) -> None:
        super().__init__()

        self.dim = dim
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * self.dim, self.dim), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), dim=2)
        return self.project(features)


# Ignore fusion embedding, just pass image embeddings through
class IgnoreFusionEmbedding(nn.Module):
    def __init__(self, start_index=1) -> None:
        super().__init__()

        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


# Reassemble image from gotten transformer unit outputs
class Reassemble(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        out_channels,
        resample: nn.Module,
        readout: nn.Module,
        coeff: int = 1,
    ) -> None:
        super().__init__()

        # Self all variables
        self.coeff = coeff
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # This conv represents projection in Reassemble block inside the transformer
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.resample = resample  # Resample function to use (a bit more flexible than defining single function here)
        self.readout = readout

    def forward(self, x: torch.Tensor):
        # AddReadout goes here, if we use exposure embedding
        x = self.readout(x)
        # Transpose image to go from (B, P, C) to (B, C, H, W)
        x = x.transpose(1, 2)
        B, C, P = x.size()
        x = x.view(
            B,
            C,
            self.image_size[0] // (self.patch_size * self.coeff),
            self.image_size[1] // (self.patch_size * self.coeff),
        )
        x = self.conv.forward(x)
        x = self.resample.forward(x)
        return x


# Residual convolution block used in upscale fusion
class ResidualConv(nn.Module):
    def __init__(
        self,
        channels,
        padding,
        pad_fn: nn.Module,
        activation: nn.Module,
        bn: bool = False,
    ) -> None:
        super().__init__()

        self.sequence = nn.Sequential(
            activation,
            pad_fn(padding),
            nn.Conv2d(channels, channels, kernel_size=3, bias=not bn),
            activation,
            pad_fn(padding),
            nn.Conv2d(channels, channels, kernel_size=3, bias=not bn),
        )
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        # Pass through sequence and do residual connection
        out = self.sequence.forward(x)
        return self.skip_add.add(out, x)


# Fusion block for mask upscale from lower parts of image
class FusionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding,
        pad_fn: nn.Module,
        activation: nn.Module,
        align_corners: bool = True,
    ) -> None:
        super().__init__()

        self.align_corners = align_corners
        self.residual_1 = ResidualConv(in_channels, padding, pad_fn, activation)
        self.residual_2 = ResidualConv(in_channels, padding, pad_fn, activation)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, *xs):
        # Since we add previous fusion with current, 1st fusion won't have anything to sum with
        out = xs[0]
        if len(xs) == 2:
            res = self.residual_1(xs[1])
            out = self.skip_add.add(out, res)

        out = self.residual_2(out)
        out = F.interpolate(
            out, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        out = self.out_conv(out)
        return out


# Patch merge block for cascading patches
class PatchMerge(nn.Module):
    def __init__(self, dim, coeff, coeff_pool, image_size, patch_size, readout) -> None:
        super().__init__()

        # Self some variables, so we can flex them
        self.coeff = coeff
        self.coeff_pool = coeff_pool
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.readout = readout

        t = 2 * self.coeff_pool
        self.rearrange = Rearrange(
            "b c (h t1) (w t2) -> b (h w) (c t1 t2)",
            t1=t,
            t2=t,
        )
        self.luma_linear = torch.nn.Linear(dim * t**2, dim)
        self.chroma_linear = torch.nn.Linear(dim * t**2, dim)

    def forward(self, l: torch.Tensor, c: torch.Tensor):
        # Handle readout token, so we don't merge it in image
        if self.readout != "identity":
            l_readout, l = l[:, 0], l[:, 1:]
            c_readout, c = c[:, 0], c[:, 1:]

        # Reassemble image
        l = l.transpose(1, 2)
        B, C, P = l.size()
        l = l.view(
            B,
            C,
            self.image_size[0] // (self.patch_size * self.coeff),
            self.image_size[1] // (self.patch_size * self.coeff),
        )
        c = c.transpose(1, 2)
        B, C, P = c.size()
        c = c.view(
            B,
            C,
            self.image_size[0] // (self.patch_size * self.coeff),
            self.image_size[1] // (self.patch_size * self.coeff),
        )

        # Here I want to do patch merging
        l = self.rearrange.forward(l)
        l = self.luma_linear.forward(l)
        c = self.rearrange.forward(c)
        c = self.chroma_linear.forward(c)

        # Cat readout token back to sequence start
        if self.readout != "identity":
            l = torch.cat([l_readout.unsqueeze(1), l], dim=1)
            c = torch.cat([c_readout.unsqueeze(1), c], dim=1)

        return l, c


# Transformer class definition - unfortunately, if I want to pass these out, I can't do very dynamic stuff
# That's a lie, I can register forward hooks, but I am too lazy to do that.
# But, I want to do this dense stuff, so I really can't do it. L I guess...
# So everything is separated and written out specifically. Manual adjustments have to be made in order to change parameters
class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        image_size: tuple,
        patch_size: int,
        readout: str,
        unit: nn.Module,
        use_patch_merge: bool,
        use_layerscale: bool,
    ) -> None:
        super().__init__()

        assert dim % heads == 0, "Dimensions must be dividable by heads!"

        # Save image and patch sizes for advanced memery for this module
        self.image_size = image_size
        self.patch_size = patch_size
        self.readout = readout

        # Patch merge layers defined - we will reuse them (that might be a mistake, but I don't care)
        if use_patch_merge:
            self.patch_merge_1_1 = PatchMerge(
                dim, 1, 1, self.image_size, self.patch_size, self.readout
            )
            self.patch_merge_2_1 = PatchMerge(
                dim, 2, 1, self.image_size, self.patch_size, self.readout
            )
            self.patch_merge_4_1 = PatchMerge(
                dim, 4, 1, self.image_size, self.patch_size, self.readout
            )
        else:
            self.patch_merge_1_1 = Identity()
            self.patch_merge_2_1 = Identity()
            self.patch_merge_4_1 = Identity()

        # Transformer layers
        self.t_layer_1 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_2 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_3 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_4 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_5 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_6 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_7 = unit(dim, heads, mlp_dim, use_layerscale)
        self.t_layer_8 = unit(dim, heads, mlp_dim, use_layerscale)

        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, luma, chroma):
        # New stuff starts from here - also going to try simple cascade without dense connections
        l1, c1 = self.t_layer_1.forward(luma, chroma)
        l2, c2 = self.t_layer_2.forward(l1, c1)
        l2_merged, c2_merged = self.patch_merge_1_1(l2, c2)

        l3, c3 = self.t_layer_3.forward(l2_merged, c2_merged)
        l4, c4 = self.t_layer_4.forward(l3, c3)
        l4_merged, c4_merged = self.patch_merge_2_1(l4, c4)

        l5, c5 = self.t_layer_5.forward(l4_merged, c4_merged)
        l6, c6 = self.t_layer_6.forward(l5, c5)
        l6_merged, c6_merged = self.patch_merge_4_1(l6, c6)

        l7, c7 = self.t_layer_7.forward(l6_merged, c6_merged)
        l8, c8 = self.t_layer_8.forward(l7, c7)

        return (l2, c2), (l4, c4), (l6, c6), (l8, c8)


# DCU based enhance skip connection (cause I am retarded)
class DCUSkipConnection(nn.Module):
    def __init__(
        self,
        img_channels,
        code_channels,
        kernel_size,
        pad: int,
        pad_fn: nn.Module,
        layers: int,
    ) -> None:
        super().__init__()

        self.code_channels = code_channels
        self.layers = layers
        self.block = nn.ModuleList(
            [
                DCU(img_channels, code_channels, kernel_size, pad, pad_fn)
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        B, _, H, W = x.size()

        # Initialize code
        code = torch.zeros(B, self.code_channels, H, W).to(x.device).to(x.dtype)
        for layer in self.block:
            code = layer.forward(x, code)

        return code


# DCU based model head for reconstruction and stuff
# Going to start with 4 heads
class DCUHead(nn.Module):
    def __init__(
        self, img_channels, code_channels, kernel_size, pad: int, pad_fn: nn.Module
    ) -> None:
        super().__init__()

        self.conv_dcu_1 = DCU(img_channels, code_channels, kernel_size, pad, pad_fn)
        self.conv_dcu_2 = DCU(img_channels, code_channels, kernel_size, pad, pad_fn)
        self.conv_dcu_3 = DCU(img_channels, code_channels, kernel_size, pad, pad_fn)
        self.conv_dcu_4 = DCU(img_channels, code_channels, kernel_size, pad, pad_fn)

        self.conv_final = nn.Conv2d(code_channels, img_channels, kernel_size=1)

    # Code is model output and data is input images
    def forward(self, data, code):
        code = self.conv_dcu_1.forward(data, code)
        code = self.conv_dcu_2.forward(data, code)
        code = self.conv_dcu_3.forward(data, code)
        code = self.conv_dcu_4.forward(data, code)
        code = self.conv_final(code)
        return code


# Straight enhance skip connection, with a lesser receptive field, but also, better scaling and stuff
class EnhanceSkipConnectionStraight(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        pad_fn: nn.Module,
        layers: int,
        use_bottleneck: bool = False,
    ) -> None:
        super().__init__()

        self.use_bottleneck = use_bottleneck

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=False),
            nn.GroupNorm(channels_out // 4, channels_out),
            nn.ReLU(True),
        )

        self.block = nn.Sequential(
            pad_fn(1),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, bias=False),
            nn.GroupNorm(channels_out // 4, channels_out),
            nn.ReLU(True),
        )
        for _ in range(1, layers):
            self.block.append(pad_fn(1))
            self.block.append(
                nn.Conv2d(channels_out, channels_out, kernel_size=3, bias=False)
            )
            self.block.append(nn.GroupNorm(channels_out // 4, channels_out))
            self.block.append(nn.ReLU(True))

    def forward(self, x):
        B, C, H, W = x.size()
        out = self.block(x)
        # Use bottleneck if necessary
        if self.use_bottleneck:
            out = out + self.bottleneck.forward(x)
        # x = F.adaptive_avg_pool2d(x, (H // 2, W // 2))
        return out


# Reassemble coupling module (going to try wrapping reassembles, so it can be used easier)
class ReassembleContainer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        readout: str,
        coeff: int = 1,
        use_patch_merge: bool = False,
    ) -> None:
        super().__init__()

        # Misc definitions, so our solution doesn't die
        self.dim = dim

        # Going to try define kernel sizes based on patch size. Otherwise it gets fairly complicated
        # I know that 16 is a magic number, you'll just have to trust me
        kernel_size = patch_size / 16
        kernels = []
        for _ in range(4):
            kernels.append(int(kernel_size))
            kernel_size *= 2

        # Define reassembles here - not really flexible, which is kinda mid, but whatever. We stopped being flexible some time ago
        self.reassemble_1 = Reassemble(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=dim,
            out_channels=32,
            resample=torch.nn.ConvTranspose2d(32, 32, kernels[3], kernels[3]),
            readout=self.get_readout_operation(readout),
            coeff=coeff,
        )
        self.reassemble_2 = Reassemble(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=dim,
            out_channels=64,
            resample=torch.nn.ConvTranspose2d(64, 64, kernels[2], kernels[2]),
            readout=self.get_readout_operation(readout),
            coeff=coeff,
        )
        self.reassemble_3 = Reassemble(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=dim,
            out_channels=128,
            resample=torch.nn.ConvTranspose2d(128, 128, kernels[1], kernels[1]),
            readout=self.get_readout_operation(readout),
            coeff=coeff,
        )
        self.reassemble_4 = Reassemble(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=dim,
            out_channels=256,
            resample=torch.nn.ConvTranspose2d(256, 256, kernels[0], kernels[0]),
            readout=self.get_readout_operation(readout),
            coeff=coeff,
        )

    # Function for getting readout operation
    def get_readout_operation(self, op: str):
        assert op in [
            "add",
            "cat",
            "ignore",
            "identity",
        ], "Allowed functions - add, cat, ignore, identity. All rest are bad"
        if op == "add":
            return AddFusionEmbedding()
        if op == "cat":
            return CatFusionEmbedding(self.dim)
        if op == "ignore":
            return IgnoreFusionEmbedding()
        if op == "identity":
            return nn.Identity()

    def forward(self, r1, r2, r3, r4):
        r1 = self.reassemble_1(r1)
        r2 = self.reassemble_2(r2)
        r3 = self.reassemble_3(r3)
        r4 = self.reassemble_4(r4)
        return r1, r2, r3, r4


# Fusion coupling module (also wrapping fusions, so they're easier to use)
class FusionContainer(nn.Module):
    def __init__(self, padding: int, pad_fn: nn.Module) -> None:
        super().__init__()

        # Define fusion blocks here
        self.fusion_1 = FusionBlock(32, 32, padding, pad_fn, nn.ReLU(True))
        self.fusion_2 = FusionBlock(64, 32, padding, pad_fn, nn.ReLU(True))
        self.fusion_3 = FusionBlock(128, 64, padding, pad_fn, nn.ReLU(True))
        self.fusion_4 = FusionBlock(256, 128, padding, pad_fn, nn.ReLU(True))

    def forward(self, r1, r2, r3, r4):
        # Do fusion
        r4_fusion = self.fusion_4(r4)
        r3_fusion = self.fusion_3(r3, r4_fusion)
        r2_fusion = self.fusion_2(r2, r3_fusion)
        r1_fusion = self.fusion_1(r1, r2_fusion)
        return r1_fusion


# Vision transformer, which includes regular transformer and rest of transformations necessary for work
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: tuple,
        patch_size: int,
        dim: int,
        heads: int,
        mlp_dim: int,
        unit: nn.Module,
        channels: int = 1,
        padding: int = 1,
        enhance_layers: int = 5,
        pad_fn: nn.Module = nn.ReflectionPad2d,  # Going to try using reflection padding for still images instead of constant
        readout: str = "identity",
        use_patch_merge: bool = True,  # Patch merge use inside model (retains basic idea)
        use_layerscale: bool = False,
    ) -> None:
        super().__init__()

        # Assert that image size must be divisible by patch size
        assert (
            image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0
        ), "Image size must be divisible by patch size. Adjust either or, but this rule must pass"

        # self parameters which could be used later on
        self.dim = dim
        self.channels = channels
        self.patch_dim = channels * patch_size**2
        self.readout = readout
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

        # Define everything connected to embeddings:
        # Embedding count based on patch count in image
        # Embeddings themselves (with dimension either flat patch or dedicated dim)
        # Rearrange operation to transform patches to embeddings (and project them as necessary)
        # Exposure embedding, which could potentially contain global information about the image
        self.num_embeddings = (image_size[0] // patch_size) * (
            image_size[1] // patch_size
        )
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=dim if dim != self.patch_dim else self.patch_dim,
        )
        # This is fixed to 3 expositions, so, if dynamic exposure count is implemented, then we're fucked
        # Also, doesn't really allow flex in dynamic exposure count. Would need to disable this if that were the case
        self.exp_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=dim if dim != self.patch_dim else self.patch_dim,
        )
        # To patch embedding by linear projection
        self.luma_to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_size**2, dim),
        )
        self.chroma_to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_size**2 * 2, dim),
        )

        # Define transformer itself
        self.transformer = Transformer(
            dim,
            heads,
            mlp_dim,
            image_size,
            patch_size,
            readout,
            unit,
            use_patch_merge,
            use_layerscale,
        )

        # Define reassemble operations for processing transformer outputs
        # These parameters check out only for patch size 16 (1, 2, 4, 8)
        if use_patch_merge:
            coeff_1, coeff_2, coeff_3, coeff_4 = 1, 2, 4, 8
            kernel_size_1 = kernel_size_2 = kernel_size_3 = kernel_size_4 = 8
        else:
            coeff_1 = coeff_2 = coeff_3 = coeff_4 = 1
            kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4 = 16, 8, 4, 2

        self.luma_reassemble = ReassembleContainer(
            image_size, patch_size, dim, self.readout
        )
        self.chroma_reassemble = ReassembleContainer(
            image_size, patch_size, dim, self.readout
        )

        # Define Fusion stuff
        self.luma_fusion = FusionContainer(padding, pad_fn)
        self.chroma_fusion = FusionContainer(padding, pad_fn)

        # Define skip connection bottleneck, trash, meme to check out how well it works
        self.luma_skip_enhance = EnhanceSkipConnectionStraight(
            channels_in=1,
            channels_out=32,
            pad_fn=pad_fn,
            layers=enhance_layers,
            use_bottleneck=False,
        )
        self.chroma_skip_enhance = EnhanceSkipConnectionStraight(
            channels_in=2,
            channels_out=32,
            pad_fn=pad_fn,
            layers=enhance_layers,
            use_bottleneck=False,
        )

        # self.luma_skip_enhance = DCUSkipConnection(
        #     img_channels=1,
        #     code_channels=32,
        #     kernel_size=3,
        #     pad=padding,
        #     pad_fn=pad_fn,
        #     layers=enhance_layers,
        # )
        # self.chroma_skip_enhance = DCUSkipConnection(
        #     img_channels=2,
        #     code_channels=32,
        #     kernel_size=3,
        #     pad=padding,
        #     pad_fn=pad_fn,
        #     layers=enhance_layers,
        # )

        # Define final output head, which would produce segmented masks used further
        # self.luma_head = nn.Sequential(
        #     pad_fn(padding),
        #     nn.Conv2d(32, 32, kernel_size=3, bias=False),
        #     nn.GroupNorm(num_groups=2, num_channels=32),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1, False),
        #     nn.Conv2d(32, 1, kernel_size=1),
        #     # Interpolate(scale_factor=2, mode="bilinear"),
        # )
        # self.chroma_head = nn.Sequential(
        #     pad_fn(padding),
        #     nn.Conv2d(32, 32, kernel_size=3, bias=False),
        #     nn.GroupNorm(num_groups=2, num_channels=32),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1, False),
        #     nn.Conv2d(32, 2, kernel_size=1),
        #     # Interpolate(scale_factor=2, mode="bilinear"),
        # )

        # TODO - Test this :)
        self.luma_head = DCUHead(
            img_channels=1,
            code_channels=32,
            kernel_size=3,
            pad=padding,
            pad_fn=pad_fn,
        )
        self.chroma_head = DCUHead(
            img_channels=2,
            code_channels=32,
            kernel_size=3,
            pad=padding,
            pad_fn=pad_fn,
        )

    def forward(self, luma: torch.Tensor, chroma: torch.Tensor):
        B, C, H, W = luma.size()

        # Work with positional embeddings
        pos_embedding = self.embedding(
            torch.arange(self.num_embeddings).repeat(B, 1).to(luma.device)
        )
        luma_emb = self.luma_to_patch_embedding(luma)
        luma_emb = self.skip_add.add(luma_emb, pos_embedding)

        # Sharing positional embedding, so we create a stronger relation to where things are w.r.t. channels
        chroma_emb = self.chroma_to_patch_embedding(chroma)
        chroma_emb = self.skip_add.add(chroma_emb, pos_embedding)

        # If we use exposure embedding, we have to add it here
        # Comment these 2 lines out, if you're not using exposure embedding (and set self.readout to "identity" before run)
        if self.readout != "identity":
            exp_embedding = self.exp_embedding(
                torch.arange(B).to(luma.device)
            ).unsqueeze(1)
            luma_emb = torch.cat([exp_embedding, luma_emb], dim=1)
            chroma_emb = torch.cat([exp_embedding, chroma_emb], dim=1)

        luma_skip = self.luma_skip_enhance(luma)
        chroma_skip = self.chroma_skip_enhance(chroma)

        # Transformer outputs
        (
            (luma_r1, chroma_r1),
            (luma_r2, chroma_r2),
            (luma_r3, chroma_r3),
            (luma_r4, chroma_r4),
        ) = self.transformer(luma_emb, chroma_emb)

        # Reassemble pseudo-images from gotten transformer outputs
        luma_r1, luma_r2, luma_r3, luma_r4 = self.luma_reassemble(
            luma_r1, luma_r2, luma_r3, luma_r4
        )
        chroma_r1, chroma_r2, chroma_r3, chroma_r4 = self.chroma_reassemble(
            chroma_r1, chroma_r2, chroma_r3, chroma_r4
        )

        # Fuse together reassembled images
        luma_out = self.luma_fusion(luma_r1, luma_r2, luma_r3, luma_r4)
        chroma_out = self.chroma_fusion(chroma_r1, chroma_r2, chroma_r3, chroma_r4)

        # Added skip connection from enhance block, which gets some local features, so we can get detailed maps out (maybe)
        # luma_out = self.luma_head(self.skip_add.add(luma_out, luma_skip))
        # chroma_out = self.chroma_head(self.skip_add.add(chroma_out, chroma_skip))

        # Skip connection for DCU head
        luma_out = self.luma_head(luma, self.skip_add.add(luma_out, luma_skip))
        chroma_out = self.chroma_head(
            chroma, self.skip_add.add(chroma_out, chroma_skip)
        )
        # DCU head without skip connection being present. This is a setup for next experiment
        # luma_out = self.luma_head(luma, luma_out)
        # chroma_out = self.chroma_head(chroma, chroma_out)

        # Return transformer as well, if necessary, otherwise don't (only final layer)
        return luma_out, chroma_out


# Model itself definition
class Model(nn.Module):
    def __init__(
        self,
        image_size: tuple,
        dim: int,
        heads: int,
        mlp_dim: int,
        padding: int,
        pad_fn: nn.Module,
        enhance_layers: int = 5,
        readout: str = "identity",
        radius=1,
        eps=1e-4,
        is_guided=True,
    ) -> None:
        super().__init__()

        self.low_res_net = VisionTransformer(
            image_size=image_size,
            patch_size=32,
            dim=dim,
            heads=heads,
            mlp_dim=mlp_dim,
            unit=FullParallelTransformerUnit,
            padding=padding,
            enhance_layers=enhance_layers,
            pad_fn=pad_fn,
            readout=readout,
            use_patch_merge=False,
            use_layerscale=False,
        )

        self.is_guided = is_guided
        if is_guided:
            # self.guided_filter = FastGuidedFilter(radius, eps)
            # TODO - Test this :)
            # self.guided_filter = ScuffedGuidedFilter(pad=padding, pad_fn=pad_fn)
            self.guided_filter_y = ConvGuidedFilter(pad_fn=pad_fn)
            self.guided_filter_cb = ConvGuidedFilter(pad_fn=pad_fn)
            self.guided_filter_cr = ConvGuidedFilter(pad_fn=pad_fn)

    # This is the regular forward pass, I'll be using for fusion - replace inputs with x_hr, when converting to ONNX
    def forward(
        self, y_lr, y_hr, cb_lr, cb_hr, cr_lr, cr_hr
    ):  # y_lr, y_hr, cb_lr, cb_hr, cr_lr, cr_hr
        # Inference conversion lines
        # x_hr = x_hr.squeeze(0)  # Dimension must be set
        # x_lr = F.interpolate(x_hr, (960, 1280), mode="bilinear")
        # y_lr, cb_lr, cr_lr = x_lr.split(1, dim=1)
        # y_hr, cb_hr, cr_hr = x_hr.split(1, dim=1)

        chroma = torch.cat([cb_lr, cr_lr], dim=1)
        w_lr, w_chroma = self.low_res_net.forward(y_lr, chroma)
        w_lr_cb, w_lr_cr = w_chroma.split(1, dim=1)

        if self.is_guided:
            w_hr = self.guided_filter_y(y_lr, w_lr, y_hr)
            w_hr_cb = self.guided_filter_cb(cb_lr, w_lr_cb, cb_hr)
            w_hr_cr = self.guided_filter_cr(cr_lr, w_lr_cr, cr_hr)
        else:
            w_hr = F.interpolate(w_lr, y_hr.size()[2:], mode="bilinear")
            w_hr_cb = F.interpolate(w_lr_cb, cb_hr.size()[2:], mode="bilinear")
            w_hr_cr = F.interpolate(w_lr_cr, cb_hr.size()[2:], mode="bilinear")

        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + 1e-8) / torch.sum(w_hr + 1e-8, dim=0)
        o_hr = torch.sum(w_hr * y_hr, dim=0, keepdim=True)

        w_hr_cb = torch.abs(w_hr_cb)
        w_hr_cb = (w_hr_cb + 1e-8) / torch.sum(w_hr_cb + 1e-8, dim=0)
        o_hr_cb = torch.sum(w_hr_cb * cb_hr, dim=0, keepdim=True)

        w_hr_cr = torch.abs(w_hr_cr)
        w_hr_cr = (w_hr_cr + 1e-8) / torch.sum(w_hr_cr + 1e-8, dim=0)
        o_hr_cr = torch.sum(w_hr_cr * cr_hr, dim=0, keepdim=True)

        # Use first return, when training and second when converting models
        return o_hr, o_hr_cb, o_hr_cr, w_hr, w_hr_cb, w_hr_cr
        # return torch.cat([o_hr, o_hr_cr, o_hr_cb], dim=1)


if __name__ == "__main__":
    device = torch.device("cuda:1")

    y_lr, cb_lr, cr_lr = torch.rand(3, 3, 768, 1536).to(device).tensor_split(3, 1)
    y_hr, cb_hr, cr_hr = torch.rand(3, 3, 1536, 3072).to(device).tensor_split(3, 1)

    model = Model(
        image_size=tuple(y_lr.size()[2:]),
        dim=512,
        heads=8,
        mlp_dim=2048,
        padding=1,
        enhance_layers=3,
        pad_fn=nn.ReflectionPad2d,
        readout="identity",
    ).to(device)

    o_hr, o_hr_cb, o_hr_cr, w_hr, w_hr_cb, w_hr_cr = model.forward(
        y_lr, y_hr, cb_lr, cb_hr, cr_lr, cr_hr
    )
    print("meme")

    pass
