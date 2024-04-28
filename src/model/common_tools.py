import torch.nn as nn

# --------------------------------------------------------------------------------
def conv_weight_init(module):
    """ Initialise Conv2D and ConvTranspose2D with N(0, 0.02)
    """
    if any(isinstance(module, m) for m in [nn.Conv2d, nn.ConvTranspose2d]):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
