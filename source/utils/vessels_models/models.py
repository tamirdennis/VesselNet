import torch
import torch.nn as nn
from torchvision import models
from .convnext_edited import create_custom_convnext
from timm.models.vision_transformer import VisionTransformer


DEVICE_IDS = list(range(torch.cuda.device_count()))


def create_custom_vit(image_size):
    """
    Builds a custom Vision Transformer with patch_size=8 and embed_dim=512
    using timm's VisionTransformer class. This allows arbitrary input sizes
    (specified by image_size), rather than a fixed 224x224.
    """

    # Create a custom ViT with the desired configuration:
    vit_model = VisionTransformer(
        img_size=image_size,
        patch_size=8,        # <--- 8x8 patches
        in_chans=3,
        num_classes=0,       # removes the classification head, yields feature embeddings
        embed_dim=512,       # <--- final feature dimension 512
        depth=12,            # same number of blocks as 'base' ViT
        num_heads=8,         # chosen so 512 is divisible by the number of heads (8)
        qkv_bias=True,
    )
    # Assign an attribute so your code knows the output dimension
    vit_model.output_channels = 512
    return vit_model

# ------------------
# ConvNeXt Classes
# ------------------
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


# ------------------
# ViT Classes
# ------------------
class VesselsMILViTABS(nn.Module):
    """
    Abstract base class for a Vessel MIL (Multiple Instance Learning) model using Vision Transformer (ViT).
    """

    def __init__(self, image_size, bag_size, in_channels=3, args=None):
        super().__init__()
        self.bag_size = bag_size
        # We won't reshape each image to "conv_next_view_shape" like in ConvNeXt,
        # but we do note the shape in case you want custom resizing logic:
        self.vit_view_shape = (in_channels, *image_size)

        self.vit_encoder = create_custom_vit(image_size)
        dim = self.vit_encoder.output_channels
        self.to_head_shape = (bag_size, dim)
        self.vit_encoder = torch.nn.DataParallel(self.vit_encoder, device_ids=DEVICE_IDS)

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


class VesselsMILViT(VesselsMILViTABS):
    """
    A standard Vessels MIL model using ViT without thickness input.
    """

    def __init__(self, image_size, bag_size, in_channels=3, args=None):
        super().__init__(image_size, bag_size, in_channels, args)

    def forward(self, x):
        """
        Forward pass for a batch of bags [batch_size, bag_size, channels, height, width].
        """
        bs = x.shape[0]
        # Reshape: (bs * bag_size, in_channels, height, width)
        x = x.view(bs * self.bag_size, *self.vit_view_shape)

        # Forward through ViT
        x = self.vit_encoder(x)  # shape => [bs*bag_size, patch_embed_dim]
        # Some ViT variants produce features at last layer => shape [bs*bag_size, D]

        # Reshape to [bs, bag_size, D]
        x = x.view(bs, *self.to_head_shape)
        # Flatten to [bs, bag_size*D]
        x = x.flatten(1)
        # MLP head
        x = self.mlp_head(x)
        return x


class VesselsWithThicknessMILViT(VesselsMILViTABS):
    """
    Vessels MIL Vision Transformer model that incorporates vessel thickness as an additional input.
    Similar logic to how we handle thickness in the ConvNeXt with multi-head attn, if needed.
    """

    def __init__(self, image_size, bag_size, in_channels=3, args=None):
        super().__init__(image_size, bag_size, in_channels, args)
        # For the thickness logic, we might define an MHA or linear. Below is a minimal approach:
        self.thickness_integration = nn.Linear(1, 32)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.vit_encoder.module.output_channels,
                                                    num_heads=2,
                                                    kdim=32,
                                                    vdim=32)

    def forward(self, x, thickness):
        bs = x.shape[0]
        x = x.view(bs * self.bag_size, *self.vit_view_shape)
        thickness = thickness.view(bs * self.bag_size, 1)

        # Step 1: Vit features
        x = self.vit_encoder(x)  # shape => [bs*bag_size, D]

        # Step 2: Reshape to [1, bs, bag_size, D] if we want an MHA approach
        x = x.view(bs, self.bag_size, -1).permute(1, 0, 2)  # => [bag_size, bs, D]

        # Step 3: Integrate thickness
        thickness_features = torch.relu(self.thickness_integration(thickness))
        # Reshape thickness to match multihead shape => [bag_size, bs, D_th]
        thickness_features = thickness_features.view(bs, self.bag_size, -1).permute(1, 0, 2)

        # Step 4: MHA
        # query = x, key = thickness, value = thickness
        attn_output, _ = self.multihead_attn(query=x, key=thickness_features, value=thickness_features)
        # For simplicity, keep the original x (you could incorporate attn_output)

        # Flatten
        attn_output = attn_output.permute(1, 0, 2)  # => [bs, bag_size, D]
        attn_output = attn_output.reshape(bs, -1)   # => [bs, bag_size*D]

        # MLP head
        x = self.mlp_head(attn_output)
        return x


def get_vessels_model(args):
    """
    Factory function to create and return a vessel model based on arguments.

    We check args.model_type:
        - 'convnext': use VesselsMILConvNext / VesselsWithThicknessMILConvNext
        - 'vit': use VesselsMILViT / VesselsWithThicknessMILViT
        If not specified, default to ConvNeXt-based model.

    Args:
        args (Namespace):
            - model_type (str): 'convnext' or 'vit'.
            - num_frames (int): The height dimension of each vessel image (if random_crop is not used).
            - random_crop (tuple or None): If not None, overrides image_size with random crop dimension.
            - use_thickness (bool): Whether to use thickness or not.
            - bag_size (int): Number of vessel samples in one bag.
            - dropout_p (float): Dropout probability.
            - etc.

    Returns:
        nn.Module: The constructed Vessels MIL model (ViT or ConvNeXt, thickness or not).
    """
    # Determine final image size
    image_size = (args.num_frames, args.vessels_length) if args.random_crop is None else args.random_crop

    model_type = getattr(args, 'model_type', 'convnext').lower()

    if model_type == 'vit':
        if args.use_thickness:
            vessels_model = VesselsWithThicknessMILViT(
                image_size=image_size,
                bag_size=args.bag_size,
                in_channels=3,
                args=args
            )
        else:
            vessels_model = VesselsMILViT(
                image_size=image_size,
                bag_size=args.bag_size,
                in_channels=3,
                args=args
            )
    else:
        # Default: convnext
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
