"""
A torch Dataset class for loading monet and photos data
"""

import cv2
import random
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------
class MonetDataset(Dataset):
    """ Monet Dataset class that deals with photos and monet data
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 root_path):
        """ Init Monet Dataset
        """

        # Paths
        self.root_path = Path(root_path)
        self.monet_paths = list((self.root_path / "monet_jpg").iterdir())
        self.photo_paths = list((self.root_path / "photo_jpg").iterdir())

    # --------------------------------------------------------------------------------
    def __len__(self):
        """ Len of dataset is equal to number of photos

        ---
        Returns
            int: Length of dataset

        """

        return len(self.photo_paths)

    # --------------------------------------------------------------------------------
    def __getitem__(self, idx):
        """ Get photo at idx and a random monet

        ---
        Parameters
            idx: Photo idx

        ---
        Returns
            tuple: 2x torch.Tensors, a random Monet and a Photo at idx

        """
        # Get a photo path
        path_photo = self.photo_paths[idx]

        # Get a random monet path
        path_monet = random.choices(self.monet_paths, k=1)[0]

        # Load imgs and transform
        x_photo = self.to_tensor(self.read_img(path_photo))
        x_monet = self.to_tensor(self.read_img(path_monet))

        return x_monet, x_photo

    # --------------------------------------------------------------------------------
    def get_sample(self,
                   idx=0,
                   as_tensor=True,
                   sample_type="monet"):
        """ Retrieves either Monet or Photo sample

        ---
        Parameters
            idx: Sample idx
            sample_type: Can be either monet or photo
            as_tensor: True will return a 3D torch.Tensor [C, H, W] (-1,1)
                else a cv2 RGB image (numpy array) [H, W, C] (0, 255)

        ---
        Returns
            torch.Tensor | numpy.ndarray: A single sample as torch or numpy

        """

        # Get path to sample
        if sample_type == "monet":
            sample_path = self.monet_paths[idx]
        elif sample_type == "photo":
            sample_path = self.photo_paths[idx]
        else:
            raise ValueError(f"Sample type can be either monet or photo, given: {sample_type}")

        # Read
        sample = self.read_img(path=sample_path)

        # Return as a 3D torch.Tensor [C, H, W]
        if as_tensor:
            return self.to_tensor(x=sample)

        # Return as a 3D numpy array
        return sample

    # --------------------------------------------------------------------------------
    def get_samples(self,
                    as_tensor=True,
                    batch_size=4,
                    sample_type="monet"):
        """ Retrieves all or N (batch_size) random samples of either monet or photo

        ---
        Parameters
            as_tensor: True will return 4D torch.Tensor [N, C, H, W] (-1,1)
                else numpy.ndarray [N, H, W, C] (0,255)
            batch_size: If not None will return a random batch of n requested samples
            sample_type: Can be either monet or photo

        ---
        Returns
            torch.Tensor | numpy.ndarray: All samples as a single Tensor
        """

        # Get paths of type
        if sample_type == "monet":
            sample_paths = self.monet_paths
        elif sample_type == "photo":
            sample_paths = self.photo_paths
        else:
            raise ValueError(f"Sample type can be either monet or photo, given: {sample_type}")

        # Check if batch was given
        if batch_size is not None:
            # Try to randomly sample
            try:
                sample_paths = random.sample(sample_paths, batch_size)
            # Negative or too large
            except ValueError as e:
                raise ValueError(f"{e}. Number of samples: {len(sample_paths)}")

        # Read sampels
        monet_samples = [self.read_img(path=path) for path in sample_paths]

        # Return as a 4D torch.Tensor
        if as_tensor:
            return torch.stack([self.to_tensor(t) for t in monet_samples], dim=0)

        # Return as a 4D numpy.ndarray
        return np.array(monet_samples)

    # --------------------------------------------------------------------------------
    def to_tensor(self, x):
        """ Transform uint8 image [0,255] to torch float32 [-1,1]

        ---
        Parameters
            x: Img tensor as a np array

        ---
        Returns
            torch.Tensor: Normalised img torch.Tensor
        """

        return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0

    # --------------------------------------------------------------------------------
    def read_img(self, path):
        """ Read img with cv2 and transform to RGB

        ---
        Parameters
            path: A path to an img

        ---
        Returns
            np.array: A RGB image
        """

        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
