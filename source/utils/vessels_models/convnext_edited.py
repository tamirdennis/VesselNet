import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from typing import Callable, List, Optional
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, Permute


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            block_kernel=7,
    ) -> None:
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
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    def __init__(
            self,
            input_channels: int,
            out_channels: Optional[int],
            num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers


class ConvNeXtVesselEncoderABS(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.1,
            layer_scale: float = 1e-6,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            stem_kernel=4,
            stem_stride=4,
            block_kernel=7,
    ) -> None:
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
            ))

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
        self.output_channels = block_setting[-1].out_channels if block_setting[-1].out_channels is not None else \
            block_setting[-1].input_channels


class ConvNeXtVesselEncoder(ConvNeXtVesselEncoderABS):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.1,
            layer_scale: float = 1e-6,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            stem_kernel=4,
            stem_stride=4,
            block_kernel=7,
    ) -> None:
        super().__init__(block_setting, stochastic_depth_prob, layer_scale, block, norm_layer, stem_kernel, stem_stride,
                         block_kernel)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x


class ConvNeXtVesselWithThicknessEncoder(ConvNeXtVesselEncoderABS):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.1,
            layer_scale: float = 1e-6,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            stem_kernel=4,
            stem_stride=4,
            block_kernel=7,
    ) -> None:
        super().__init__(block_setting, stochastic_depth_prob, layer_scale, block, norm_layer, stem_kernel, stem_stride,
                         block_kernel)
        self.thickness_integration = nn.Linear(1, 32)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.output_channels, num_heads=2, kdim=32, vdim=32)

    def forward(self, x: Tensor, thickness) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)[:, :, 0, 0].unsqueeze(0)
        thickness_features = F.relu(self.thickness_integration(thickness)).unsqueeze(0)
        attn_output, _ = self.multihead_attn(query=x, key=thickness_features, value=thickness_features)
        return x


class ConvNeXtVesselRegressor(ConvNeXtVesselEncoder):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.1,
            layer_scale: float = 1e-6,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block_setting, stochastic_depth_prob, layer_scale, block, norm_layer)
        lastblock = block_setting[-1]
        lastconv_output_channels = lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        self.head = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.head(x)
        return x


def create_custom_convnext(args, use_thickness=False):
    convnext_base_params = 8 * args.num_params_multiplier
    custom_block_setting = [
        CNBlockConfig(convnext_base_params, 2 * convnext_base_params, 2),
        CNBlockConfig(2 * convnext_base_params, 4 * convnext_base_params, 2),
        CNBlockConfig(4 * convnext_base_params, 8 * convnext_base_params, 6),
        CNBlockConfig(8 * convnext_base_params, None, 2)
    ]
    if use_thickness:
        model = ConvNeXtVesselWithThicknessEncoder(custom_block_setting,
                                                   stem_kernel=args.stem_kernel, stem_stride=args.stem_stride,
                                                   block_kernel=args.block_kernel,
                                                   )

    else:
        model = ConvNeXtVesselEncoder(custom_block_setting,
                                      stem_kernel=args.stem_kernel, stem_stride=args.stem_stride,
                                      block_kernel=args.block_kernel,
                                      )
    return model
