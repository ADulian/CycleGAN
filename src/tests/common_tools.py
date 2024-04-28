""" Unit tests for the common_tools
"""
import pytest
import torch.nn as nn

from src.model.common_tools import conv_weight_init

# --------------------------------------------------------------------------------
def test_conv_weight_init():
    """
    Tests the conv_weight_init function.

    Verifies that the function initializes Conv2D and ConvTranspose2D modules
    with a normal distribution (mean 0, std 0.02) and sets the bias to 0.
    """

    # Create a Conv2D module
    conv2d = nn.Conv2d(3, 6, kernel_size=3)
    conv_weight_init(conv2d)
    assert conv2d.weight.mean().item() == pytest.approx(0.0, abs=1e-2)
    assert conv2d.weight.std().item() == pytest.approx(0.02, abs=1e-2)
    assert conv2d.bias is not None
    assert conv2d.bias.mean().item() == 0.0

    # Create a ConvTranspose2D module
    conv_transpose2d = nn.ConvTranspose2d(3, 6, kernel_size=3)
    conv_weight_init(conv_transpose2d)
    assert conv_transpose2d.weight.mean().item() == pytest.approx(0.0, abs=1e-2)
    assert conv_transpose2d.weight.std().item() == pytest.approx(0.02, abs=1e-2)
    assert conv_transpose2d.bias is not None
    assert conv_transpose2d.bias.mean().item() == 0.0
