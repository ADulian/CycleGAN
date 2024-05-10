"""
A module with the Discriminator class

"""

import random
import torch
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

        # --- History
        self.history = DiscriminatorHistory(batch_size=4)

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
                x,
                sample_history=False):
        """ Discriminate between real/fake img

        ---
        Parameters
            x: Input tensor (img)
            sample_history: If true it will mix x with samples from buffer

        ---
        Returns
            torch.Tensor: Output tensor (pred grid)

        """

        # Update x based on history
        if self.history is not None:
            x = self.history(x)


        return self.discriminator(x)

# --------------------------------------------------------------------------------
class DiscriminatorHistory:
    """ Keep a history of fake samples that Discriminator can use to improve its robustness

    The history has a buffer size B that is set to N*Batch Size, as the N increases so does
    the variation of samples, initially I guess that a larger variation might be useful but
    as the model learns it is probably safe to keep it smaller

    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 batch_size,
                 n=2):
        """ Init History module"""

        self.sample_size = batch_size // 2
        self.buffer_size = batch_size * n
        self.data_indices = list(range(0, self.buffer_size))

        # Empty Tensor
        self.buffer = torch.Tensor()


    # --------------------------------------------------------------------------------
    def __call__(self,
                 x):
        """ Forward

        Sample randomly from history buffer and add new sampels

        ---
        Parameters
            x: Input Tensor

        ---
        Returns
            Tensor with mixed samples from input and history

        """

        # Make sure that the buffer is on the same device as x
        if self.buffer.device != x.device:
            self.buffer = self.buffer.to(x.device)

        # Check if buffer is filled
        if len(self.buffer) >= self.buffer_size:
            # Sample from the buffer
            history_indices = random.sample(self.data_indices, k=self.sample_size)
            x1 = self.buffer[history_indices]

            # Sample from the input
            x_indices = list(range(0, len(x)))
            x_get_indices = random.sample(x_indices, k=self.sample_size)
            x_set_indices = list(set(x_indices) - set(x_get_indices))
            x2 = x[x_get_indices]

            # Replace buffer with other set of samples
            self.buffer[history_indices] = x[x_set_indices]

            # Concat
            x = torch.cat((x1, x2), 0)

        # Fill the buffer
        else:
            self.buffer = torch.cat((self.buffer, x), 0)

        # Return
        return x
