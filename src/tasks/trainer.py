"""
Trainer class used to manage the training process of the model
"""

import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.model.cycle_gan import CycleGAN
from src.dataset.monet_dataset import MonetDataset

# --------------------------------------------------------------------------------
class Trainer:
    """ Trainer manages the training process and anything related to it (e.g. loading/saving weights)
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 data_root,
                 output_path=None,
                 batch_size=32,
                 device=None):
        """ Initialise the Trainer class

        ---
        Parameters
            data_root: Path to dataset
            output_path: Path where outputs will be saved
            batch_size: Data loader batch size
            device: Device, if None then CPU will be used

        """

        # Path
        self.output_path = Path("outputs") if output_path is None else Path(output_path)
        if not self.output_path.exists():  # By default save in the wd
            self.output_path.mkdir()

        # On CPU by  default
        self.device = torch.device("cpu") if device is None else device

        # Init Model, Dataset and Data Loader
        # Probably bit of an overkill but might be useful to have access to those in notebooks
        self.model = CycleGAN()
        self.dataset = MonetDataset(root_path=data_root)
        self.data_loader = DataLoader(dataset=self.dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=4)

    # --------------------------------------------------------------------------------
    def fit(self,
            num_epochs=1,
            save_weights=True,
            load_weights_path=None):
        """ Fit the model on dataset

        ---
        Parameters
            num_epochs: Number of trianing epochs
            save_weights: Whether to save weights after training is finished
            load_weights_path: Load weights from given path

        """

        # --- Model Setup
        self.model = self.model.to(self.device)
        self.model.setup()
        if load_weights_path is not None:
            self.load_weights(load_path=load_weights_path)

        # --- Train
        loss_epoch = defaultdict(list)
        for i in range(1, num_epochs + 1):  # Starting at one for tqdm

            # Tqdm
            with tqdm(total=len(self.data_loader),
                      desc=f"Epoch {i}/{num_epochs}",
                      ascii=True,
                      colour="green",
                      dynamic_ncols=True) as pbar:

                # Per batch
                loss_batch = defaultdict(list)
                for x in self.data_loader:
                    x_monet = x[0].to(self.device)
                    x_photo = x[1].to(self.device)

                    # Batch of data
                    loss = self.model.forward_step((x_monet, x_photo))

                    # Update dict
                    for k, v in loss.items():
                        loss_batch[k].append(v.item())

                    # Bar Update
                    pbar.set_postfix({k: v.item() for k, v in loss.items()})
                    pbar.update()

                # Update mean loss for epoch
                for k, v in loss_batch.items():
                    loss_epoch[k].append(np.mean(v))

        # Save weights of the final model
        if save_weights:
            self.save_weights()

    # --------------------------------------------------------------------------------
    def save_weights(self,
                     save_path=None):
        """ Save model weights

        ---
        Parameters
            save_path: Save path for weights
        """

        # Default path is output_path
        if save_path is None:
            save_path = self.output_path

        save_path = Path(save_path) / "model.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Weights saved at: {save_path}")

    # --------------------------------------------------------------------------------
    def load_weights(self,
                     load_path):
        """ Load model weights

        ---
        Parameters
            load_path: Load path for weights

        """

        self.model.load_state_dict(torch.load(load_path))
        print(f"Weights loaded from: {load_path}")
