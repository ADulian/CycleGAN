""" An extension of the Image Explorer tool made specifically for visualisation of CycleGANs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

from src.tools.image_explorer import ImageExplorer


# --------------------------------------------------------------------------------
class GANExplorer(ImageExplorer):
    """ Overrides some ImageExplorer functionality to make it easy to explore the GAN
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 model,
                 dataset,
                 batch_size=8,
                 style_b="monet",
                 render_discriminator=True,
                 *args, **kwargs):
        """ Init Gan Explorer

        ---
        Parameters
            model: Cycle GAN
            dataset: Monet-Photos Dataset
            batch_size: Buffer of samples per single inference run
            style_b: Style (Domain) B, i.e. the one that is being applied to Domain A
            render_discriminator: If True it will render output of discriminator too (heatmap)
        """
        super().__init__(dataset=dataset, num_samples=batch_size,
                         *args, **kwargs)

        self.model = model
        self.model_output = None
        self.style_b = style_b
        self.style_a = "monet" if style_b == "photo" else "photo"
        self.max_samples = len(dataset.monet_paths) if style_b == "photo" else len(dataset.photo_paths)
        self.batch_size = min(batch_size, self.max_samples)
        self.total_images = self.n_rows
        self.dataset_prev_index = 0
        self.dataset_next_index = 0
        self.render_discriminator = render_discriminator

        # Update col number of figsize
        if self.render_discriminator:
            self.n_cols *= 2
            self.figsize =  (self.figsize[0] * 2, self.figsize[1])

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
    def _update_display(self):
        """ Update figure
        """

        # If current index is beyond/above min/max run inference
        if self.current_index < 0:
            self._get_prev_samples()
        elif self.current_index > (self.num_samples - self.total_images) \
                or self.model_output is None:
            self._get_next_samples()

        with self.output_widget:
            clear_output(wait=True)
            self._display_images()

    # --------------------------------------------------------------------------------
    def _display_images(self):
        """ Display GAN output in the following format

        Single row = data from a single sample
            C1 - Real A
            C2 - Fake B
            C3 - Cycled A

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
                real_A, fake_B, cycled_A, disc_real_A, disc_fake_B, disc_cycled_A = self._get_model_sample(idx=idx)
                gan_output = [real_A, fake_B, cycled_A,
                              disc_real_A, disc_fake_B, disc_cycled_A]

                for j in range(self.n_cols):
                    axs[i, j].imshow(gan_output[j])
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])

                    if self.render_discriminator:
                        if j >= self.n_cols // 2:
                            axs[i, j].set_xlabel(f"Discriminator Mean: {gan_output[j].mean():.2f}",
                                                 fontsize=12)
                    else:
                        axs[i, j].set_xlabel(f"Discriminator Mean: {gan_output[j + self.n_cols].mean():.2f}",
                                             fontsize=12)

        # Set shared titles
        titles = ["Real A", "Fake B", "Cycled A",
                  "Disc Real A (\u2191)", "Disc Fake B (\u2193)", "Disc Cycled A (\u2191)"]
        for i in range(self.n_cols):
            axs[0, i].set_title(titles[i])

        # Set row titles just as ylabel
        for i in range(self.n_rows):
            axs[i, 0].set_ylabel(f"Sample idx: {self.dataset_prev_index + i + 1 + self.current_index}",
                                 fontsize=12)

        plt.show()

    # --------------------------------------------------------------------------------
    def _get_model_sample(self, idx):
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

        # Check if model output exists
        if self.model_output is None:
            raise ValueError("Model Output is None, run inference first")

        # Set to last idx
        if idx >= self.num_samples:
            idx = self.num_samples - 1

        # Get output of gen
        real_A = self._to_numpy(self.model_output["real_A"][idx])
        fake_B = self._to_numpy(self.model_output["gen_fake_B"][idx])
        cycled_A = self._to_numpy(self.model_output["gen_cycled_A"][idx])

        # And disc
        disc_real_A = self._to_numpy(self.model_output["disc_real_A"][idx])
        disc_fake_B = self._to_numpy(self.model_output["disc_fake_B"][idx])
        disc_cycled_A = self._to_numpy(self.model_output["disc_cycled_A"][idx])

        return real_A, fake_B, cycled_A, disc_real_A, disc_fake_B, disc_cycled_A

    # --------------------------------------------------------------------------------
    def _get_prev_samples(self):
        """ Get previous samples from dataset (n=batch_size) and run inference
        """
        # Beyond first set of samples
        if self.dataset_prev_index < 0:
            self.current_index = 0
        else:
            # Get n samples
            samples = []
            for i in range(self.batch_size):

                samples.append(self.dataset.get_sample(idx=self.dataset_prev_index,
                                                       as_tensor=True,
                                                       sample_type=self.style_a))
                self.dataset_prev_index -= 1
                if self.dataset_prev_index < 0:
                    break

            samples = torch.stack(samples[::-1], dim=0)

            # Inference
            model_output = self.model.forward_apply_style(samples,
                                                          style=self.style_b)

            # Merge
            num_next_elements = self.n_rows - 1
            if num_next_elements > 0:  # Only a case if n_rows == 1
                for k in self.model_output:
                    next_output = self.model_output[k][:num_next_elements]
                    self.model_output[k] = torch.cat((model_output[k], next_output))
                self.current_index = len(samples) + num_next_elements - self.n_rows
            else:
                self.model_output = model_output
                self.current_index = 0

            # Update
            self.num_samples = len(samples) + num_next_elements
            self.dataset_next_index = self.dataset_prev_index + self.num_samples + 1

    # --------------------------------------------------------------------------------
    def _get_next_samples(self):
        """ Get next samples from dataset (n=batch_size) and run inference
        """

        # End of dataset
        if self.dataset_next_index == self.max_samples:
            self.current_index = self.num_samples - self.n_rows
        else:
            # Get N samples
            samples = []
            for i in range(self.batch_size):
                samples.append(self.dataset.get_sample(idx=self.dataset_next_index,
                                                       as_tensor=True,
                                                       sample_type=self.style_a))
                self.dataset_next_index += 1
                if self.dataset_next_index == self.max_samples:
                    break

            # Stack
            samples = torch.stack(samples, dim=0)

            # Inference
            model_output = self.model.forward_apply_style(samples,
                                                          style=self.style_b)

            # Initial entry
            if self.model_output is None:
                self.model_output = model_output
                self.num_samples = len(samples)
            # If not None, merge output of n_rows - 1
            else:
                num_prev_elements = self.n_rows - 1
                if num_prev_elements > 0:
                    for k in self.model_output:
                        prev_output = self.model_output[k][-num_prev_elements:]
                        self.model_output[k] = torch.cat((prev_output, model_output[k]))
                else:
                    self.model_output = model_output

                self.num_samples = len(samples) + num_prev_elements

            # Update indices
            self.dataset_prev_index = self.dataset_next_index - self.num_samples - 1
            self.current_index = 0

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
