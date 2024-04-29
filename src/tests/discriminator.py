"""
Unit tests for Discriminator
"""

import pytest
import torch
from src.model.discriminator import Discriminator

# --------------------------------------------------------------------------------
def test_discriminator_init():
    """ Test the initialization of the Discriminator
    """
    discriminator = Discriminator()
    assert isinstance(discriminator, torch.nn.Module)

# --------------------------------------------------------------------------------
def test_discriminator_forward():
    """ Test the forward pass of the Discriminator
    """
    discriminator = Discriminator()
    input_tensor = torch.randn(1, 3, 256, 256)  # Create a random input tensor
    output = discriminator(input_tensor)
    assert output.shape == torch.Size([1, 1, 30, 30])  # Check the output shape
