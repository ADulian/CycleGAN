"""
A module with the Discriminator class

"""

import torch.nn as nn

from src.model.common_blocks import Downsample
from src.model.common_tools import conv_weight_init

# --------------------------------------------------------------------------------
class Discriminator(nn.Module):
    """ A Discriminator Network

    ---
    Structure
        Img -> Discriminator -> Grid of predictions

    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """ Init CycleGAN's Discriminator

        """

        super().__init__()

        # --- Default settings
        base_channels = 64
        kernel_size = 4

        # --- Discriminator network
        self.discriminator = nn.Sequential(
            Downsample(in_channels=3, out_channels=base_channels,
                       kernel_size=kernel_size, norm=False),
            Downsample(in_channels=base_channels, out_channels=base_channels * 2,
                       kernel_size=kernel_size),
            Downsample(in_channels=base_channels * 2, out_channels=base_channels * 4,
                       kernel_size=kernel_size),
            nn.ZeroPad2d(padding=(0, 2, 0, 2)),
            nn.Conv2d(in_channels=base_channels * 4, out_channels=base_channels * 8,
                      kernel_size=4, stride=1, bias=False),
            nn.InstanceNorm2d(num_features=base_channels * 8),
            nn.ZeroPad2d(padding=(0, 2, 0, 2)),
            nn.Conv2d(in_channels=base_channels * 8, out_channels=1,
                      kernel_size=4, stride=1),
            #             nn.Sigmoid()
        )

        # Initialise conv wights with N(0, 0.02)
        self.apply(conv_weight_init)

    # --------------------------------------------------------------------------------
    def forward(self,
                x):
        """ Discriminate between real/fake img

        ---
        Parameters
            x: Input tensor (img)

        ---
        Returns
            torch.Tensor: Output tensor (pred grid)

        """
        return self.discriminator(x)
