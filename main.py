from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from visualize import reconstruct

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_updates", type=int, default=250000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=10)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--dataset", type=str, default='MNIST')

# whether or not to save model
parser.add_argument("--filename",  type=str, default='model')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
samples = [validation_data[i][0] for i in range(16)]
samples_plots_path = Path('tmp/vqvae_samples')
samples_plots_path.mkdir(exist_ok=True, parents=True)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

channels = samples[0].shape[0]
args.__dict__['channels'] = channels

model = VQVAE(channels, args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def save_model(n_updates):
    hyperparameters = args.__dict__
    utils.save_model_and_results(
        model, results, hyperparameters, args.filename)

    print('Update #', n_updates, 'Recon Error:',
          np.mean(results["recon_errors"][-args.log_interval:]),
          'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
          'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


def train():
    n_updates = 0
    while True:
        for x, _ in training_loader:
            n_updates += 1
            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = n_updates

            if n_updates % args.log_interval == 0:
                """
                save model and print values
                """
                save_model(n_updates)
                reconstruct(model, samples, device, str(samples_plots_path / f'step{n_updates}'))

            if n_updates > args.n_updates:
                save_model(n_updates)
                return


if __name__ == "__main__":
    train()
