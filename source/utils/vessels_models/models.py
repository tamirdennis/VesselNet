import torch
import torch.nn as nn
from .convnext_edited import create_custom_convnext


DEVICE_IDS = list(range(torch.cuda.device_count()))


class VesselsMILConvNextABS(nn.Module):
    """
    Abstract base class for a Vessel MIL (Multiple Instance Learning) model using ConvNeXt.
    """

    def __init__(self, image_size, bag_size, in_channels=3, args=None):
        """
        Initializes the abstract base class for the VesselsMILConvNext model.

        Args:
            image_size (tuple): The expected (height, width) of each vessel image sample.
            bag_size (int): Number of samples (vessels) in one bag.
            in_channels (int, optional): Number of input channels for the vessel image. Default=3.
            args (Namespace, optional): Parsed arguments containing model hyperparameters.
        """
        super().__init__()
        self.bag_size = bag_size
        self.conv_next_view_shape = (in_channels, *image_size)

        self.conv_next_encoder = create_custom_convnext(args, use_thickness=False)
        dim = self.conv_next_encoder.output_channels
        self.to_head_shape = (bag_size, dim)
        self.conv_next_encoder = torch.nn.DataParallel(self.conv_next_encoder, device_ids=DEVICE_IDS)

        self.fc_dim = dim * bag_size

        if args and args.dropout_p > 0:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.fc_dim, eps=1e-6),
                nn.Linear(self.fc_dim, self.fc_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=args.dropout_p),
                nn.Linear(self.fc_dim // 2, 1),
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.fc_dim, eps=1e-6),
                nn.Linear(self.fc_dim, self.fc_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fc_dim // 2, 1),
            )


class VesselsMILConvNext(VesselsMILConvNextABS):
    """
    A standard Vessels MIL model using ConvNeXt without thickness input.
    """

    def __init__(self, image_size, bag_size, in_channels=3, args=None):
        """
        Inherit from VesselsMILConvNextABS but keep thickness disabled.

        Args:
            image_size (tuple): The (height, width) of each vessel image sample.
            bag_size (int): Number of vessel samples in one bag.
            in_channels (int, optional): Number of channels in the image. Default=3.
            args (Namespace, optional): Contains model hyperparams, e.g., dropout_p.
        """
        super().__init__(image_size, bag_size, in_channels, args)

    def forward(self, x):
        """
        Forward pass for a batch of bags.

        Args:
            x (torch.Tensor): Shape [batch_size, bag_size, in_channels, height, width].

        Returns:
            torch.Tensor: Shape [batch_size, 1], the predicted regression value (e.g., HGB).
        """
        bs = x.shape[0]
        x = x.view((bs * self.bag_size, *self.conv_next_view_shape))
        x = self.conv_next_encoder(x)
        x = x.view((bs, *self.to_head_shape))
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x


class VesselsWithThicknessMILConvNext(VesselsMILConvNextABS):
    """
    Vessels MIL ConvNeXt model that incorporates vessel thickness as an additional input.
    """

    def __init__(self, image_size, bag_size, in_channels=3, args=None):
        """
        Initializes the MIL model with thickness input.

        Args:
            image_size (tuple): The (height, width) of each vessel image sample.
            bag_size (int): Number of vessel samples in one bag.
            in_channels (int, optional): Number of channels in the image. Default=3.
            args (Namespace, optional): Contains model hyperparams, e.g., dropout_p.
        """
        super().__init__(image_size, bag_size, in_channels, args)
        self.conv_next_encoder = create_custom_convnext(args, use_thickness=True)
        self.conv_next_encoder = torch.nn.DataParallel(self.conv_next_encoder, device_ids=DEVICE_IDS)

    def forward(self, x, thickness):
        """
        Forward pass for a batch of bags with thickness input.

        Args:
            x (torch.Tensor): Shape [batch_size, bag_size, in_channels, height, width].
            thickness (torch.Tensor): Shape [batch_size, bag_size, 1] with thickness values.

        Returns:
            torch.Tensor: Shape [batch_size, 1], the predicted regression value (e.g., HGB).
        """
        bs = x.shape[0]
        x = x.view((bs * self.bag_size, *self.conv_next_view_shape))
        thickness = thickness.view((bs * self.bag_size, 1))

        x = self.conv_next_encoder(x, thickness)
        x = x.view((bs, *self.to_head_shape))
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x


def get_vessels_model(args):
    """
    Factory function to create and return a vessel model based on arguments.

    Args:
        args (Namespace):
            - num_frames (int): The height dimension of each vessel image if random_crop is not used.
            - random_crop (tuple or None): If not None, overrides image_size with random crop dimension.
            - use_thickness (bool): Whether to use thickness or not.
            - bag_size (int): Number of vessel samples in one bag.
            - (Other ConvNeXt-related args: stem_kernel, stem_stride, block_kernel, etc.)

    Returns:
        nn.Module: The constructed Vessels MIL model (with or without thickness).
    """
    image_size = (args.num_frames, args.vessels_length) if args.random_crop is None else args.random_crop
    if args.use_thickness:
        vessels_model = VesselsWithThicknessMILConvNext(
            image_size=image_size,
            bag_size=args.bag_size,
            in_channels=3,
            args=args
        )
    else:
        vessels_model = VesselsMILConvNext(
            image_size=image_size,
            bag_size=args.bag_size,
            in_channels=3,
            args=args
        )

    vessels_model = vessels_model.cuda()
    return vessels_model
