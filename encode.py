import argparse
from pathlib import Path
from typing import Union
import torch
import numpy as np

from utils import load_data_and_data_loaders
from utils import load_vqvae


def encode(model_path: Union[str, Path], output_path: Union[str, Path]):
    model_path = Path(model_path)
    output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae, _ = load_vqvae(model_path, device)

    encodings_batches = []
    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders('CIFAR10', 128)
    for (x, _) in validation_loader:
        x = x.to(device)
        encodings = vqvae.encode(x)
        encodings_batches.append(encodings)

    all_encoding = torch.unsqueeze(torch.cat(encodings_batches, 0), 1).cpu().numpy()
    np.save(output_path, all_encoding)


def main():
    parser = argparse.ArgumentParser("Encode whole CIFAR-10 dataset using selected model.")
    parser.add_argument("model", help="Model path")
    parser.add_argument("output", help='Output file')
    args = parser.parse_args()

    encode(args.model, args.output)


if __name__ == '__main__':
    main()
