"""
Main entrypoint, for now just training
"""
import argparse
import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.cycle_gan import CycleGAN
from src.dataset.monet_dataset import MonetDataset

# --------------------------------------------------------------------------------
def train(args):
    """ Train the model
    """

    # --- Params (Will put them later into argparse
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- Model and Dataset
    print("Initialising Model and Dataset")
    cycle_gan = CycleGAN().to(device)
    cycle_gan.setup()

    dataset = MonetDataset(root_path=args.data_root)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=4)

    # --- Train
    print("Training")
    loss_epoch = defaultdict(list)
    for i in range(1, NUM_EPOCHS + 1):  # Starting at one for tqdm

        # Tqdm
        with tqdm(total=len(data_loader),
                  desc=f"Epoch {i}/{NUM_EPOCHS}",
                  ascii=True,
                  colour="green",
                  dynamic_ncols=True) as pbar:

            # Per batch
            loss_batch = defaultdict(list)
            for x in data_loader:
                x_monet = x[0].to(device)
                x_photo = x[1].to(device)

                # Batch of data
                loss = cycle_gan.forward_step((x_monet, x_photo))

                # Update dict
                for k, v in loss.items():
                    loss_batch[k].append(v.item())

                # Bar Update
                pbar.set_postfix({k: v.item() for k, v in loss.items()})
                pbar.update()

            # Update mean loss for epoch
            for k, v in loss_batch.items():
                loss_epoch[k].append(np.mean(v))

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # --- Args
    parser = argparse.ArgumentParser(description='Simple implementation of CycleGAN in PyTorch')
    parser.add_argument('--data_root', type=str, default="data/",
                        help='Root path to dataset')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing data (default: 32)')

    args = parser.parse_args()

    # --- Train
    train(args)

