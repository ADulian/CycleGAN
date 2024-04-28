""" Unit tests for the Downsample class
"""

import torch
import torch.nn as nn

from src.model.common import Downsample, Upsample

class TestDownsample:
    """ Unit tests for the Downsample class
    """

    def test_init(self):
        """ Test that the Downsample class is initialised correctly

        Verifies that the Downsample class is initialised with the correct attributes
        """

        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3, norm=True)
        assert isinstance(downsample, nn.Module)
        assert downsample.conv.kernel_size[0] == 3
        assert downsample.norm.num_features == 64

    def test_forward(self):
        """Test that the forward method downsamples the input tensor correctly

        Verifies that the forward method downsamples the input tensor by 2
        """

        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = downsample(input_tensor)
        assert output.shape == (1, 64, 128, 128)

    def test_forward_no_norm(self):
        """Test that the forward method works correctly when norm=False

        Verifies that the forward method works correctly when instance normalisation is disabled
        """

        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3, norm=False)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = downsample(input_tensor)
        assert output.shape == (1, 64, 128, 128)

    def test_conv_layer(self):
        """
        Test that the convolutional layer is configured correctly

        Verifies that the convolutional layer has the correct stride, padding, and bias
        """
        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3)
        assert isinstance(downsample.conv, nn.Conv2d)
        assert downsample.conv.stride == (2, 2)
        assert downsample.conv.padding == (1, 1)

    def test_instance_norm_layer(self):
        """Test that the instance normalisation layer is configured correctly

        Verifies that the instance normalisation layer has the correct number of features
        """

        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3, norm=True)
        assert isinstance(downsample.norm, nn.InstanceNorm2d)
        assert downsample.norm.num_features == 64

        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3, norm=False)
        assert downsample.norm is None

    def test_activation_layer(self):
        """Test that the activation layer is configured correctly

        Verifies that the activation layer is a LeakyReLU layer
        """

        downsample = Downsample(in_channels=3, out_channels=64, kernel_size=3)
        assert isinstance(downsample.act, nn.LeakyReLU)

class TestUpsample:
    def test_init(self):
        upsample = Upsample(in_channels=3, out_channels=64, kernel_size=3, dropout=True)
        assert isinstance(upsample, nn.Module)
        assert upsample.conv.kernel_size[0] == 3
        assert upsample.norm.num_features == 64
        assert upsample.dropout.p == 0.5 if upsample.dropout else None

    def test_forward(self):
        upsample = Upsample(in_channels=3, out_channels=64)
        input_tensor = torch.randn(1, 3, 128, 128)
        output = upsample(input_tensor)
        assert output.shape == (1, 64, 256, 256)

    def test_forward_no_dropout(self):
        upsample = Upsample(in_channels=3, out_channels=64, dropout=False)
        input_tensor = torch.randn(1, 3, 128, 128)
        output = upsample(input_tensor)
        assert output.shape == (1, 64, 256, 256)

    def test_conv_layer(self):
        upsample = Upsample(in_channels=3, out_channels=64, kernel_size=3)
        assert isinstance(upsample.conv, nn.ConvTranspose2d)
        assert upsample.conv.stride == (2, 2)
        assert upsample.conv.padding == (1, 1)

    def test_instance_norm_layer(self):
        upsample = Upsample(in_channels=3, out_channels=64, kernel_size=3)
        assert isinstance(upsample.norm, nn.InstanceNorm2d)
        assert upsample.norm.num_features == 64

    def test_activation_layer(self):
        upsample = Upsample(in_channels=3, out_channels=64, kernel_size=3)
        assert isinstance(upsample.act, nn.ReLU)
