"""
A module with the Generator class

"""

import torch
import torch.nn as nn

from src.model.common_blocks import Downsample, Upsample
from src.model.common_tools import conv_weight_init

# --------------------------------------------------------------------------------
class Generator(nn.Module):
    """ A Generator Network

    ---
    Structure
        Img -> Encoder -> Decoder -> Img_Pred

        Encoder
            7 x Downsample blocks
            Output -> Latent Tensor [512, IMG_H / 128, IMG_W / 128]
        Decoder
            6 x Upsample blocks
            Output -> Upsampled Latent Tensor [64, IMG_H / 2, IMG_W / 2]
        Final Upsample
            ConvTranspose2D -> TanH
            Output -> Target Image [IMG_C, IMG_H, IMG_W]

    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """ Init CycleGAN's Generator

        """
        super().__init__()

        # Encoder and Decoder
        self.encoder = self._init_encoder()
        self.decoder = self._init_decoder()

        # Final Upsample
        self.out = nn.ConvTranspose2d(in_channels=128,
                                      out_channels=3,
                                      kernel_size=4,
                                      stride=2, padding=1)

        # Maybe softsign could be better than TanH
        self.act = nn.Tanh()

        # Initialise conv wights with N(0, 0.02)
        self.apply(conv_weight_init)

    # --------------------------------------------------------------------------------
    def forward(self,
                x):
        """ Generate x_hat

        ---
        Parameters
            x: Input tensor (img)

        ---
        Returns
            torch.Tensor: Output tensor (img pred)

        """

        # Encode to latent space / 128 #512
        skips = []  # Skip connections U-Net Style
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # Skip last one, bottom bit, latent space
        skips = reversed(skips[:-1])

        # Decode form latent space *64 #128
        for layer, skip in zip(self.decoder, skips):
            x = layer(x)
            x = torch.cat((x, skip), dim=1)

        # Upsample so that out.shape == x.shape
        out = self.act(self.out(x))

        return out

    # --------------------------------------------------------------------------------
    def _init_encoder(self):
        """ A Sequential Encoder with 7 Downsample Blocks each downsampling by 2

        ---
        Returns
            nn.Sequential: Encoder

        """

        # Settings
        kernel_size = 4
        in_channels = 64

        # Final layer will have 512 feature maps and it's final output size is x / 128
        out_channels = [128, 256, 512, 512, 512, 512]

        # First layer doesn't use norm so just add it now
        encoder = [Downsample(in_channels=3, out_channels=in_channels,
                              kernel_size=kernel_size, norm=False)]

        for out_ch in out_channels:
            # Add layer
            encoder.append(Downsample(in_channels=in_channels, out_channels=out_ch,
                                      kernel_size=kernel_size))

            # Update in_channels
            in_channels = out_ch

        return nn.Sequential(*encoder)

    # --------------------------------------------------------------------------------
    def _init_decoder(self):
        """ A Sequential Decoder with 6 Upsample Blocks each upscaling by 2

        ---
        Returns
            nn.Sequential: Decoder

        """

        # Settings
        kernel_size = 4
        in_channels = [512, 1024, 1024, 1024, 512, 256]

        # Final layer will have the same number of channels as the output of first downsample from encoder
        out_channels = [512, 512, 512, 256, 128, 64]
        dropout = [True, True, True, False, False, False]

        decoder = [Upsample(in_channels=in_ch, out_channels=out_ch,
                            kernel_size=kernel_size, dropout=drop)

                   for in_ch, out_ch, drop in zip(in_channels, out_channels, dropout)]

        return nn.Sequential(*decoder)
