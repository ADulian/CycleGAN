""" An extension of the Image Explorer tool made specifically for visualisation of CycleGANs
"""

import numpy as np
import matplotlib.pyplot as plt

from src.tools.image_explorer import ImageExplorer

# --------------------------------------------------------------------------------
class GANExplorer(ImageExplorer):
    """ Overrides some ImageExplorer functionality to make it easy to explore the GAN
    """

    # --------------------------------------------------------------------------------
    def __init__(self, model_out, *args, **kwargs):
        """ Init Gan Explorer
        """

        self.model_out = model_out

        super().__init__(num_samples=len(model_out), *args, **kwargs)

        self.total_images = self.n_rows

    # --------------------------------------------------------------------------------
    def _on_up_click(self, button):
        """ On up click
        """

        self.current_index -= 1
        self._update_display()

    # --------------------------------------------------------------------------------
    def _on_down_click(self, button):
        """ On down click
        """

        self.current_index += 1
        self._update_display()

    # --------------------------------------------------------------------------------
    def _display_images(self):
        """ Display GAN output in the following format

        Single row = data from a single sample
            C1 - Real A
            C2 - Fake B
            c3 - Cycled A

        For now it just shows the output of generator but next feature
        will involve adding the discriminator ouput in a form a heatmap

        """

        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.figsize)
        if self.n_rows == 1:
            axs = axs[np.newaxis, :]
        elif self.n_cols == 1:
            axs = axs[:, np.newaxis]

        # Get sample and put it into correct axs
        for i in range(self.n_rows):
            idx = self.current_index + i
            if idx < self.num_samples:
                real_A, fake_B, cycled_A = self._get_gan_sample(idx=idx)
                gan_output = [real_A, fake_B, cycled_A]

                assert len(gan_output) == self.n_cols

                for j in range(self.n_cols):
                    axs[i, j].imshow(gan_output[j])
                    axs[i, j].axis('off')

        # Set shared titles
        titles = ["Real A", "Fake B", "Cycled A"]
        for i in range(self.n_cols):
            axs[0, i].set_title(titles[i])

        plt.show()


    # --------------------------------------------------------------------------------
    def _get_gan_sample(self, idx):
        """ Get a sample from model's output

        ---
        Parameters
            idx (int): Index of the sample, an idx beyond actual len will be set to last item

        ---
        Returns
            tuple(np.arrays): Samples in a form of numpy arrays including
                real_A: Input to the model
                fake_B: Input transformed into domain B
                cycled_A: Input cycled back from B to A


        """

        # Set to last idx
        if idx >= self.num_samples:
            idx = self.num_samples - 1

        # Get output of gen
        real_A = self._to_numpy(self.model_out["real_A"][idx])
        fake_B = self._to_numpy(self.model_out["gen_fake_B"][idx])
        cycled_A = self._to_numpy(self.model_out["gen_cycled_A"][idx])

        return real_A, fake_B, cycled_A

    # --------------------------------------------------------------------------------
    def _to_numpy(self, torch_tensor):
        """ Transform torch tensor to numpy array

        Torch tensor is expected to be in a form of model's output i.e. [C, H, W]
        and have it's values within range of [-1, 1]

        Order of operations:
            Detach from graph (this call should be redundant if in inference mode, sanity)
            To CPU
            To Numpy
            Transpose from [C, H, W] to [H, W, C]
            Tranform range from [-1, 1] to [0, 1]

        ---
        Parameters
            torch_tensor (torch.Tensor): Torch tensor

        ---
        Returns
            np.array: Torch tensor transformed to numpy

        """
        return np.transpose(torch_tensor.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5
