"""
Trainer class used to manage the training process of the model
"""

import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.cycle_gan import CycleGAN
from src.dataset.monet_dataset import MonetDataset

# --------------------------------------------------------------------------------
class Trainer:
    """ Trainer manages the training process and anything related to it (e.g. loading/saving weights)
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 data_root,
                 batch_size=32,
                 device=None):
        """ Initialise the Trainer class

        ---
        Parameters
            data_root: Path to dataset
            batch_size: Data loader batch size
            device: Device, if None then CPU will be used

        """

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
            num_epochs=1):
        """ Fit the model on dataset

        ---
        Parameters
            data_loader: PyTorch data loader
            num_epochs: number of training epochs

        """
        # --- Model Setup
        self.model = self.model.to(self.device)
        self.model.setup()

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
