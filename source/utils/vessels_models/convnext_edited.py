import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from typing import Callable, List, Optional
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, Permute


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm for 2D data (channels_last format), but internally re-permuted for channels_first usage.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies Layer Normalization across each channel.

        Args:
            x (Tensor): Shape [B, C, H, W].

        Returns:
            Tensor: Normalized tensor with same shape [B, C, H, W].
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    """
    A ConvNeXt block with depthwise convolution and linear layers.
    """

    def __init__(self,
                 dim,
                 layer_scale: float,
                 stochastic_depth_prob: float,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 block_kernel=7):
        """
        Args:
            dim (int): Number of input channels.
            layer_scale (float): Scaling factor applied to the block output.
            stochastic_depth_prob (float): Probability for StochasticDepth.
            norm_layer (Callable[..., nn.Module], optional): Normalization layer constructor.
            block_kernel (int, optional): Depthwise conv kernel size.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=block_kernel, padding=block_kernel // 2, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the CNBlock.

        Args:
            input (Tensor): Shape [B, C, H, W].

        Returns:
            Tensor: The output of the CNBlock with the same shape [B, C, H, W].
        """
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    """
    Configuration for each stage (group of blocks) in ConvNeXt.
    """

    def __init__(self, input_channels: int, out_channels: Optional[int], num_layers: int):
        """
        Args:
            input_channels (int): Number of input channels.
            out_channels (int or None): Number of output channels after the stage, or None if there is no downsampling.
            num_layers (int): How many CNBlocks to stack.
        """
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers


class ConvNeXtVesselEncoderABS(nn.Module):
    """
    Abstract base class for a ConvNeXt-based encoder applied to vessel images.
    """

    def __init__(self,
                 block_setting: List[CNBlockConfig],
                 stochastic_depth_prob: float = 0.1,
                 layer_scale: float = 1e-6,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 stem_kernel=4,
                 stem_stride=4,
                 block_kernel=7):
        """
        Args:
            block_setting (List[CNBlockConfig]): Configuration for each stage.
            stochastic_depth_prob (float, optional): Probability for StochasticDepth.
            layer_scale (float, optional): Scaling factor for block outputs.
            block (Callable[..., nn.Module], optional): Block constructor. Defaults to CNBlock.
            norm_layer (Callable[..., nn.Module], optional): Normalization layer constructor.
            stem_kernel (int, optional): Kernel size for the initial stem convolution.
            stem_stride (int, optional): Stride for the initial stem convolution.
            block_kernel (int, optional): Kernel size for the depthwise conv in each block.
        """
        super().__init__()

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, block_kernel=block_kernel))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.output_channels = block_setting[-1].out_channels if block_setting[-1].out_channels is not None \
            else block_setting[-1].input_channels


class ConvNeXtVesselEncoder(ConvNeXtVesselEncoderABS):
    """
    Standard ConvNeXt encoder for vessel images, outputting a single pooled feature map.
    """

    def __init__(self,
                 block_setting: List[CNBlockConfig],
                 stochastic_depth_prob: float = 0.1,
                 layer_scale: float = 1e-6,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 stem_kernel=4,
                 stem_stride=4,
                 block_kernel=7):
        super().__init__(block_setting, stochastic_depth_prob, layer_scale, block, norm_layer,
                         stem_kernel, stem_stride, block_kernel)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ConvNeXt encoder followed by average pooling.

        Args:
            x (Tensor): [B, 3, H, W] input images.

        Returns:
            Tensor: [B, C] after avgpool, where C is output_channels.
        """
        x = self.features(x)
        x = self.avgpool(x)
        return x


class ConvNeXtVesselWithThicknessEncoder(ConvNeXtVesselEncoderABS):
    """
    ConvNeXt encoder variant that integrates vessel thickness via multihead attention.
    """

    def __init__(self,
                 block_setting: List[CNBlockConfig],
                 stochastic_depth_prob: float = 0.1,
                 layer_scale: float = 1e-6,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 stem_kernel=4,
                 stem_stride=4,
                 block_kernel=7):
        super().__init__(block_setting, stochastic_depth_prob, layer_scale, block, norm_layer,
                         stem_kernel, stem_stride, block_kernel)
        self.thickness_integration = nn.Linear(1, 32)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.output_channels,
            num_heads=2,
            kdim=32,
            vdim=32
        )

    def forward(self, x: Tensor, thickness) -> Tensor:
        """
        Forward pass that fuses thickness embedding via multihead attention.

        Args:
            x (Tensor): [B, 3, H, W] images.
            thickness (Tensor): [B, 1] thickness values, one per image/bag sample.

        Returns:
            Tensor: [B, 1, C] after attention, typically reshaped by the calling code.
        """
        x = self.features(x)
        x = self.avgpool(x)[:, :, 0, 0].unsqueeze(0)
        thickness_features = F.relu(self.thickness_integration(thickness)).unsqueeze(0)
        attn_output, _ = self.multihead_attn(
            query=x,
            key=thickness_features,
            value=thickness_features
        )
        return x


class ConvNeXtVesselRegressor(ConvNeXtVesselEncoder):
    """
    A ConvNeXt-based regressor that outputs a single value (e.g., HGB).
    """

    def __init__(self,
                 block_setting: List[CNBlockConfig],
                 stochastic_depth_prob: float = 0.1,
                 layer_scale: float = 1e-6,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__(block_setting, stochastic_depth_prob, layer_scale, block, norm_layer)
        lastblock = block_setting[-1]
        lastconv_output_channels = lastblock.out_channels if lastblock.out_channels is not None \
            else lastblock.input_channels
        self.head = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass that applies the base encoder and then a linear head.

        Args:
            x (Tensor): [B, 3, H, W] input.

        Returns:
            Tensor: [B, 1] regression output.
        """
        x = super().forward(x)
        x = self.head(x)
        return x


def create_custom_convnext(args, use_thickness=False):
    """
    Builds a ConvNeXt-based encoder according to user args.

    Args:
        args (Namespace):
            - stem_kernel (int): Kernel size for the initial stem.
            - stem_stride (int): Stride for the initial stem.
            - block_kernel (int): Kernel size for the depthwise convolution in blocks.
            - num_params_multiplier (int): Base multiplier for channel dimensions.
        use_thickness (bool, optional):
            If True, builds the ConvNeXtVesselWithThicknessEncoder; otherwise ConvNeXtVesselEncoder.

    Returns:
        nn.Module: A ConvNeXt-based encoder configured with the provided args.
    """
    convnext_base_params = 8 * args.num_params_multiplier
    custom_block_setting = [
        CNBlockConfig(convnext_base_params, 2 * convnext_base_params, 2),
        CNBlockConfig(2 * convnext_base_params, 4 * convnext_base_params, 2),
        CNBlockConfig(4 * convnext_base_params, 8 * convnext_base_params, 6),
        CNBlockConfig(8 * convnext_base_params, None, 2)
    ]
    if use_thickness:
        model = ConvNeXtVesselWithThicknessEncoder(
            custom_block_setting,
            stem_kernel=args.stem_kernel,
            stem_stride=args.stem_stride,
            block_kernel=args.block_kernel,
        ).cuda()
    else:
        model = ConvNeXtVesselEncoder(
            custom_block_setting,
            stem_kernel=args.stem_kernel,
            stem_stride=args.stem_stride,
            block_kernel=args.block_kernel,
        ).cuda()
    return model
