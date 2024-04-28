"""
A module that contains common building blocks for building networks

"""

import torch.nn as nn


# --------------------------------------------------------------------------------
class Downsample(nn.Module):
    """ A simple convolutional block that downsamples feature maps by 2 (stride=2)

    For example
        input_tensor [256, 256] -> output_tensor [128, 128]

    Padding = same
    Bias = False (although this will be checked)

    Model
        Conv2d -> (opt) InstanceNorm2d -> LeakyReLU

    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 norm=True):

        """ Init downsample module

        ---
        Parameters
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            norm: Whether to apply normalisation layer
        """

        super().__init__()

        # --- Layers
        # Convolution Layer
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=2,  # Downsample by 2
                              padding=(kernel_size - 1) // 2,  # Same padding
                              bias=False)  # ToDo: Check if this should be conditioned on norm

        # Normalisation layer (equivalent of GroupNorm where num_groups=out_channels
        if norm:
            self.norm = nn.InstanceNorm2d(num_features=out_channels)
        else:
            self.norm = None

        # Activation layer
        self.act = nn.LeakyReLU()

    # --------------------------------------------------------------------------------
    def forward(self,
                x):

        """ Downsample input tensor

        ---
        Parameters
            x: Input tensor

        ---
        Returns
            out: x downsampled by 2

        """

        out = self.conv(x)

        if self.norm:
            out = self.norm(out)

        out = self.act(out)

        return out


# --------------------------------------------------------------------------------
class Upsample(nn.Module):
    """ A simple convolutional block that upsamples feature maps by 2 (kernel_size=4, stride=2)

        For example
            input_tensor [128, 128] -> output_tensor [256, 256]

        Padding = same
        Bias = False
        Dropout = 0.5

        Model
            Conv2d -> InstanceNorm2d -> (opt) Dropout -> ReLU

        """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 dropout=True):
        """ Init upsample module

        ---
        Parameters
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            dropout: Whether to apply dropout
        """

        super().__init__()

        # --- Layers
        # Transpose Convolution Layer
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=2,  # Upsample by 2
                                       padding=(kernel_size - 1) // 2,  # Same padding
                                       bias=False)

        # Normalisation layer (equivalent of GroupNorm where num_groups=out_channels
        self.norm = nn.InstanceNorm2d(num_features=out_channels)

        # Dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        # Activation layer
        self.act = nn.ReLU()

    # --------------------------------------------------------------------------------
    def forward(self,
                x):
        """ Upsample input tensor

        ---
        Parameters
            x: Input tensor

        ---
        Returns
            out: x upsampled by 2

        """

        out = self.conv(x)
        out = self.norm(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.act(out)

        return out
