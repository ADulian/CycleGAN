"""
Unit tests for the Generator
"""

import torch
import torch.nn as nn

from src.model.common_blocks import Downsample, Upsample
from src.model.generator import Generator

# --------------------------------------------------------------------------------
def test_generator_init():
    """ Test that the Generator class is initialized correctly.

    Verifies that the Generator class has the expected attributes
    """

    gen = Generator()
    assert isinstance(gen, nn.Module)
    assert hasattr(gen, 'encoder')
    assert hasattr(gen, 'decoder')
    assert hasattr(gen, 'out')
    assert hasattr(gen, 'act')

# --------------------------------------------------------------------------------
def test_generator_forward():
    """ Test that the forward method produces an output tensor with the correct shape.

    Verifies that the output shape matches the input shape.
    """

    gen = Generator()
    img_shape = (1, 3, 256, 256)
    input_tensor = torch.randn(*img_shape)
    output_tensor = gen(input_tensor)
    assert output_tensor.shape == img_shape  # output shape should match input shape

# --------------------------------------------------------------------------------
def test_generator_encoder():
    """ Test that the encoder is a Sequential module with 7 Downsample blocks.

    Verifies that the encoder has the correct number of layers and that each layer is a Downsample block.
    """

    gen = Generator()
    encoder = gen.encoder

    assert isinstance(encoder, nn.Sequential)
    assert len(encoder) == 7  # 7 downsample blocks
    for layer in encoder:
        assert isinstance(layer, Downsample)

# --------------------------------------------------------------------------------
def test_generator_decoder():
    """ Test that the decoder is a Sequential module with 6 Upsample blocks.

    Verifies that the decoder has the correct number of layers and that each layer is an Upsample block.
    """

    gen = Generator()
    decoder = gen.decoder
    assert isinstance(decoder, nn.Sequential)
    assert len(decoder) == 6  # 6 upsample blocks
    for layer in decoder:
        assert isinstance(layer, Upsample)

# --------------------------------------------------------------------------------
def test_generator_out():
    """ Test that the out module is a ConvTranspose2d module with the correct parameters.

    Verifies that the out module has the correct in_channels, out_channels, kernel_size, stride, and padding.
    """

    gen = Generator()
    out = gen.out
    assert isinstance(out, nn.ConvTranspose2d)
    assert out.in_channels == 128
    assert out.out_channels == 3
    assert out.kernel_size == (4, 4)
    assert out.stride == (2, 2)
    assert out.padding == (1, 1)

# --------------------------------------------------------------------------------
def test_generator_act():
    """ Test that the act module is a Tanh module.

    Verifies that the act module is an instance of nn.Tanh.
    """

    gen = Generator()
    act = gen.act
    assert isinstance(act, nn.Tanh)