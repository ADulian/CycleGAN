import torch
import torch.nn as nn
import torch.optim as optim

from src.model.generator import Generator
from src.model.discriminator import Discriminator

# --------------------------------------------------------------------------------
class CycleGAN(nn.Module):
    """ An implementation of CycleGAN

    ---
        Structure:
        - 2 Generators
            - Monet Generator - To go from Photo to Monet (G)
            - Photo Generator - To got from Monet to Photo (F)

        - 2 Discriminators
            - Monet Discriminator - To say whether input is a real/fake monet (Dy)
            - Photo Discriminator - To say whether input is a real/fake photo (Dx)

    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 lr = 2e-4,
                 lambda_cycle = 10):
        """ Init CycleGan
        """

        super().__init__()

        # --- Generators and Discriminators
        self.gen_monet = Generator()
        self.gen_photo = Generator()

        self.disc_monet = Discriminator()
        self.disc_photo = Discriminator()

        # --- Attributes
        self.lambda_cycle = lambda_cycle  # Cycle loss weight
        self.lr = lr
        self.is_setup = False  # Checks if setup was done

        # --- Placeholders for optims and loss f
        self.optim_gen_monet = None
        self.optim_gen_photo = None

        self.optim_disc_monet = None
        self.optim_disc_photo = None

        self.loss_l2 = None
        self.loss_l1 = None


    # --------------------------------------------------------------------------------
    def forward(self, x):
        """ Main point of the model is to go from photo -> monet so forward does just that

        Expects a single tensor (photo data)
            x: [batch, 3, 256, 256] - Real photo

        ---
        Parameters
            x: Input tensor (photo img)

        ---
        Returns:
            torch.Tensor: Output tensor (photo turned to monet)

        """

        return self.gen_monet(x)

    # --------------------------------------------------------------------------------
    def forward_apply_style(self,
                            real_A,
                            style="monet"):
        """ Apply one of the styles either monet or photo

        In addition the function returns more than just styled photo e.g.
        Disciminators on Fake, Real and Cycled Photo

        ---
        Parameters
            real_A: A batch of real samples from domain A being transfered to domain B
            style: Domain B, Applies either monet or photo style

        ---
        Returns
            dict: Dictionary of outputs from model

        """

        with torch.inference_mode(mode=True):
            self.eval()


            # Set networks and Domain
            if style == "monet":
                domain_A = "photo"

                gen_A = self.gen_photo
                gen_B = self.gen_monet

                disc_A = self.disc_photo
                disc_B = self.disc_monet
            elif style == "photo":
                domain_A = "monet"

                gen_A = self.gen_photo
                gen_B = self.gen_monet

                disc_A = self.disc_photo
                disc_B = self.disc_monet
            else:
                raise ValueError(f"Style can be either monet or photo, given: {style}")

            # Style is always domain B
            domain_B = style

            # ---
            # Forward, notation uses P=Photo and M=Monet to easier demonstrate the flow
            # x_P -> G_M -> \hat{x}_M
            fake_B = gen_B(real_A)

            # \hat{x}_M -> G_P -> \hat{x}_P
            cycled_A = gen_A(fake_B)

            # \hat{x}_M -> D_M -> \hat{y}_M
            disc_fake_B = disc_B(fake_B)

            # x_P -> D_P -> y_P
            disc_real_A = disc_A(real_A)

            # \hat{x}_P -> D_M -> \hat{y}_P
            disc_cycled_A = disc_A(cycled_A)

        return {f"fake_{domain_B}" : fake_B,
                f"cycled_{domain_A}" : cycled_A,
                f"disc_fake_{domain_B}" : disc_fake_B,
                f"disc_real_{domain_A}" : disc_real_A,
                f"disc_cycled_{domain_A}" : disc_cycled_A}


    # --------------------------------------------------------------------------------
    def forward_step(self, x):
        """ A single forward step for the model

        Forward -> Loss -> Backward -> Optimise

        Expects a tuple of 2 tensors
            (tensor, tensor): [batch, 3, 256, 256] - Real monet and photo

        ---
        Parameters
            x: Tuple of 2 tensors, photo and monet img

        ---
        Returns
            dict: A dictionary of losses for Generators and Discriminators

        """

        # Check if setup was completed before proceeding
        if not self.is_setup:
            raise RuntimeError(
                "Setup function has not been called. "
                "Please call the setup function before proceeding to setup optimisers and loss functions.")

        # Unpack data
        x_monet, x_photo = x

        # Zero Grad Optimisers
        self.optim_zero_grad()

        # --- Generate fake data
        x_fake_monet = self.gen_monet(x_photo)  # Real Photo -> Gm -> Fake Monet
        x_fake_photo = self.gen_photo(x_monet)  # Real Monet -> Gp -> Fake Photo

        # --- Discriminators
        # Monet
        d_real_monet = self.disc_monet(x_monet)  # How real does this real monet seems
        d_fake_monet = self.disc_monet(x_fake_monet.detach())  # How fake does this fake seems

        # Photo
        d_real_photo = self.disc_photo(x_photo)
        d_fake_photo = self.disc_photo(x_fake_photo.detach())

        # Discriminators Loss
        d_loss_monet = self.disc_loss(x_real=d_real_monet,
                                      x_fake=d_fake_monet)

        d_loss_photo = self.disc_loss(x_real=d_real_photo,
                                      x_fake=d_fake_photo)

        d_loss = d_loss_monet + d_loss_photo

        # Backward
        d_loss.backward()

        # Optimise
        self.optim_disc_monet.step()
        self.optim_disc_photo.step()

        # --- Generators
        # Adversarial
        x_adv_monet = self.disc_monet(x_fake_monet)  # How real does this fake monet seems
        x_adv_photo = self.disc_photo(x_fake_photo)

        # Cycle
        x_cycled_monet = self.gen_monet(x_fake_photo)  # Fake Photo -> Gm -> Cycled Monet
        x_cycled_photo = self.gen_photo(x_fake_monet)  # Fake Monet -> Gp -> Cycled Photo

        # Adversarial Loss
        g_loss_adv_monet = self.loss_l2(torch.ones_like(x_adv_monet), x_adv_monet)
        g_loss_adv_photo = self.loss_l2(torch.ones_like(x_adv_photo), x_adv_photo)

        # Cycle Loss
        g_loss_cycle_monet = self.loss_l1(x_monet, x_cycled_monet) * self.lambda_cycle
        g_loss_cycle_photo = self.loss_l1(x_photo, x_cycled_photo) * self.lambda_cycle

        # Identity Loss
        # g_loss_idt_monet = self.loss_l1(x_monet, x_fake_monet) * self.lambda_cycle
        # g_loss_idt_photo = self.loss_l1(x_photo, x_fake_photo) * self.lambda_cycle

        # Total Loss
        g_loss_monet = g_loss_adv_monet + g_loss_cycle_monet  # + g_loss_idt_monet
        g_loss_photo = g_loss_adv_photo + g_loss_cycle_photo  # + g_loss_idt_photo
        g_loss = g_loss_monet + g_loss_photo

        # Backward
        g_loss.backward()

        # Optimise
        self.optim_gen_monet.step()
        self.optim_gen_photo.step()

        # Get Losses
        return {
            "g_monet": g_loss_monet,
            "g_photo": g_loss_photo,
            "d_monet": d_loss_monet,
            "d_photo": d_loss_photo
        }

    # --------------------------------------------------------------------------------
    def disc_loss(self,
                  x_real,
                  x_fake):
        """ Compute loss for discriminator

        Loss = 1/2(Real loss + Fake loss)

        ---
        Parameters
            x_real: [batch, 3, 256, 256] - A real photo
            x_fake: [batch, 3, 256, 256] - A fake photo from generator

        ---
        Returns
            torch.Tensor: An average loss of Discriminator

        """

        loss_real = self.loss_l2(torch.ones_like(x_real), x_real)
        loss_fake = self.loss_l2(torch.zeros_like(x_fake), x_fake)

        return (loss_real + loss_fake) * 0.5

    # --------------------------------------------------------------------------------
    def optim_zero_grad(self):
        """ Apply zero grad on all optimisers
        """

        # Generators
        self.optim_gen_monet.zero_grad()
        self.optim_gen_photo.zero_grad()

        # Discriminators
        self.optim_disc_monet.zero_grad()
        self.optim_disc_photo.zero_grad()

    # --------------------------------------------------------------------------------
    def setup(self):
        """ Setup optimisers and loss functions

        Optims and loss are not needed outside of training so setup just for that

        """

        # Setup Optims
        self.optim_gen_monet = optim.Adam(self.gen_monet.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.999))

        self.optim_gen_photo = optim.Adam(self.gen_photo.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.999))

        self.optim_disc_monet = optim.Adam(self.disc_monet.parameters(),
                                           lr=self.lr,
                                           betas=(0.5, 0.999))

        self.optim_disc_photo = optim.Adam(self.disc_photo.parameters(),
                                           lr=self.lr,
                                           betas=(0.5, 0.999))


        # Setup Loss
        self.loss_l2 = nn.MSELoss()
        self.loss_l1 = nn.L1Loss()

        # Update
        self.is_setup = True
