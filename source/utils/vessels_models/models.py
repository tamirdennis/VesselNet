import torch
import torch.nn as nn
from .convnext_edited import create_custom_convnext


class VesselsMILConvNextABS(nn.Module):
    def __init__(self, image_size, bag_size,
                 in_channels=3, args=None):
        super().__init__()
        self.bag_size = bag_size
        self.conv_next_view_shape = (in_channels, *image_size)

        self.conv_next_encoder = create_custom_convnext(args, use_thickness=False)
        dim = self.conv_next_encoder.output_channels
        self.to_head_shape = (bag_size, dim)
        self.conv_next_encoder = torch.nn.DataParallel(self.conv_next_encoder, device_ids=[0, 1, 2, 3])

        self.fc_dim = dim * bag_size
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.fc_dim, eps=1e-6),
            nn.Linear(self.fc_dim, self.fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_p),
            nn.Linear(self.fc_dim // 2, 1),
        ) if args.dropout_p > 0 else nn.Sequential( # no dropout
            nn.LayerNorm(self.fc_dim, eps=1e-6),
            nn.Linear(self.fc_dim, self.fc_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fc_dim // 2, 1),
        )


class VesselsMILConvNext(VesselsMILConvNextABS):
    def __init__(self, image_size, bag_size,
                 in_channels=3, args=None):
        super().__init__(image_size, bag_size, in_channels, args)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view((bs * self.bag_size, *self.conv_next_view_shape))

        x = self.conv_next_encoder(x)
        x = x.view((bs, *self.to_head_shape))

        # reshape before Fully Connected:
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x


class VesselsWithThicknessMILConvNext(VesselsMILConvNextABS):
    def __init__(self, image_size, bag_size,
                 in_channels=3, args=None):
        super().__init__(image_size, bag_size, in_channels, args)
        self.conv_next_encoder = create_custom_convnext(args, use_thickness=True)
        self.conv_next_encoder = torch.nn.DataParallel(self.conv_next_encoder, device_ids=[0, 1, 2, 3])

    def forward(self, x, thickness):
        bs = x.shape[0]
        x = x.view((bs * self.bag_size, *self.conv_next_view_shape))
        thickness = thickness.view((bs * self.bag_size, 1))

        x = self.conv_next_encoder(x, thickness)
        x = x.view((bs, *self.to_head_shape))

        # reshape before Fully Connected:
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x


def get_vessels_model(args):
    # loading vessels model:
    image_size = (args.num_frames, args.vessels_length) if args.random_crop is None else args.random_crop
    if args.use_thickness:
        vessels_model = VesselsWithThicknessMILConvNext(image_size=image_size,
                                                         bag_size=args.bag_size,
                                                         in_channels=3, args=args)
    else:
        vessels_model = VesselsMILConvNext(image_size=image_size,
                                           bag_size=args.bag_size,
                                           in_channels=3, args=args)

    vessels_model = vessels_model.cuda()
    return vessels_model
