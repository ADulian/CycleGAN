"""
Main entrypoint, for now just training
"""
import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.cycle_gan import CycleGAN
from src.dataset.monet_dataset import MonetDataset

# --------------------------------------------------------------------------------
if __name__ == '__main__':


    # --- Params (Will put them later into argparse
    NUM_EPOCHS = 1
    BATCH_SIZE = 32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- Model and Dataset
    print("Initialising Model and Dataset")
    cycle_gan = CycleGAN().to(device)
    cycle_gan.setup()

    dataset = MonetDataset(root_path="../data")
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
                  colour="green") as pbar:

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
