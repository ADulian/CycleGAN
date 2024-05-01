"""
Main entrypoint, for now just training
"""
import argparse

import torch

from src.tasks.trainer import Trainer

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # --- Args
    parser = argparse.ArgumentParser(description='Simple implementation of CycleGAN in PyTorch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', type=str, default="data/",
                        help='Root path to dataset')
    parser.add_argument('--load_subset', type=float, default=None,
                        help='Load subset of the data, either num samples (> 1) or proportion (float 0-1). '
                             'To get one sample just do 1.1 as the float is cast down to int')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,  # Max that fits onto 24GB GPU
                        help='Batch size for processing data, mine\'s set to max for 3090 24GB')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--weights_path', type=str, default=None,  # Path to load weights
                        help='Weights path, loads only if not None')
    parser.add_argument('--output_path', type=str, default=None,  # Path to saving data e.g. weights
                        help='Save all data in "." if None')
    parser.add_argument('--wandb', type=bool, default=False,
                        help='User Weights and Biases')
    parser.add_argument('--wandb_watch', type=bool, default=False,  # Watch model with wandb
                        help='Save all data in "." if None')

    args = parser.parse_args()
    if args.load_subset > 1.:
        args.load_subset = int(args.load_subset)

    # Init trainer
    device = torch.device("cuda") if torch.cuda.is_available() else None
    trainer = Trainer(device=device,
                      **vars(args))

    # Fit
    if args.wandb:
        trainer.fit_wandb(num_epochs=args.num_epochs,
                          load_weights_path=args.weights_path)
    else:
        trainer.fit(num_epochs=args.num_epochs,
                    load_weights_path=args.weights_path)
