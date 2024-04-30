"""
Main entrypoint, for now just training
"""
import argparse
import torch

from src.tasks.trainer import Trainer

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # --- Args
    parser = argparse.ArgumentParser(description='Simple implementation of CycleGAN in PyTorch')
    parser.add_argument('--data_root', type=str, default="data/",
                        help='Root path to dataset')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,  # Max that fits onto 24GB GPU
                        help='Batch size for processing data (default: 64)')

    args = parser.parse_args()

    # Init trainer
    device = torch.device("cuda") if torch.cuda.is_available() else None
    trainer = Trainer(data_root=args.data_root,
                      batch_size=args.batch_size,
                      device=device)

    # Fit
    trainer.fit(num_epochs=args.num_epochs)
