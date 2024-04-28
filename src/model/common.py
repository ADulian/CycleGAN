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

        # --- Attribs
        self.norm = norm

        # --- Layers
        # Convolution Layer
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=2,
                              padding= (kernel_size - 1) // 2,
                              bias=False)  # ToDo: Check if this should be conditioned on norm

        # Normalisation layer (equivalent of GroupNorm where num_groups=out_channels
        self.instance_norm = nn.InstanceNorm2d(num_features=out_channels)

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
            out = self.instance_norm(out)

        out = self.act(out)

        return out
