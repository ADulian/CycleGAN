"""
Unit tests for CycleGAN
"""
import pytest
import torch

from src.model.cycle_gan import CycleGAN

# --------------------------------------------------------------------------------
def test_cyclegan_init():
    """ Test the initialization of the CycleGAN
    """
    cyclegan = CycleGAN()
    assert isinstance(cyclegan, torch.nn.Module)
    assert cyclegan.lambda_cycle == 10
    assert cyclegan.lr == 2e-4
    assert not cyclegan.is_setup

# --------------------------------------------------------------------------------
def test_cyclegan_forward():
    """ Test the forward pass of the CycleGAN
    """
    cyclegan = CycleGAN()
    input_tensor = torch.randn(1, 3, 256, 256)  # Create a random input tensor
    output = cyclegan(input_tensor)
    assert output.shape == torch.Size([1, 3, 256, 256])  # Check the output shape

# --------------------------------------------------------------------------------
def test_cyclegan_forward_step():
    """ Test the forward step of the CycleGAN
    """
    cyclegan = CycleGAN()
    cyclegan.setup()
    input_tensor = (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))  # Create a random input tensor
    losses = cyclegan.forward_step(input_tensor)
    assert isinstance(losses, dict)
    assert set(losses.keys()) == {"g_monet", "g_photo", "d_monet", "d_photo"}

# --------------------------------------------------------------------------------
def test_cyclegan_forward_step_no_setup():
    """ Test the forward step of the CycleGAN without setup
    """
    cyclegan = CycleGAN()
    input_tensor = (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))  # Create a random input tensor
    with pytest.raises(RuntimeError):
        cyclegan.forward_step(input_tensor)

# --------------------------------------------------------------------------------
def test_cyclegan_disc_loss():
    """ Test the discriminator loss function
    """
    cyclegan = CycleGAN()
    cyclegan.setup()
    x_real = torch.randn(1, 3, 256, 256)  # Create a random real tensor
    x_fake = torch.randn(1, 3, 256, 256)  # Create a random fake tensor
    loss = cyclegan.disc_loss(x_real, x_fake)
    assert isinstance(loss, torch.Tensor)

# --------------------------------------------------------------------------------
def test_cyclegan_setup():
    """ Test the setup method of the CycleGAN
    """
    cyclegan = CycleGAN()
    cyclegan.setup()
    assert cyclegan.is_setup
    assert isinstance(cyclegan.optim_gen_monet, torch.optim.Adam)
    assert isinstance(cyclegan.optim_gen_photo, torch.optim.Adam)
    assert isinstance(cyclegan.optim_disc_monet, torch.optim.Adam)
    assert isinstance(cyclegan.optim_disc_photo, torch.optim.Adam)
    assert isinstance(cyclegan.loss_l2, torch.nn.MSELoss)
    assert isinstance(cyclegan.loss_l1, torch.nn.L1Loss)
