"""
A torch Dataset class for loading monet and photos data
"""

import cv2
import random
import torch

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
        path_photo = self.root_path / "photo_jpg" / self.photo_paths[idx]

        # Get a random monet path
        path_monet = self.root_path / "monet_jpg" / random.choices(self.monet_paths, k=1)[0]

        # Load imgs and transform
        x_photo = self.to_tensor(self.read_img(path_photo))
        x_monet = self.to_tensor(self.read_img(path_monet))

        return x_monet, x_photo

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
