import argparse
from pathlib import Path
from typing import Union
import torch
import numpy as np

from utils import load_data_and_data_loaders
from utils import load_vqvae


def encode_from_loader(vqvae, loader, output_path: Path, device):
    encodings_batches = []
    for (x, _) in loader:
        x = x.to(device)
        encodings = vqvae.encode(x)
        encodings_batches.append(encodings)
    all_encoding = torch.cat(encodings_batches, 0).cpu().numpy()
    np.save(output_path, all_encoding)


def encode(model_path: Union[str, Path], output_path: Union[str, Path]):
    model_path = Path(model_path)
    output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae, _ = load_vqvae(model_path, device)

    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders('CIFAR10', 128)

    print("Encoding CIFAR-10 validation...")
    encode_from_loader(vqvae, validation_loader, output_path / 'encoded_cifar10_validation.npy', device)

    print("Encoding CIFAR-10 train...")
    encode_from_loader(vqvae, training_loader, output_path / 'encoded_cifar10_train.npy', device)


def main():
    parser = argparse.ArgumentParser("Encode whole CIFAR-10 dataset using selected model.")
    parser.add_argument("model", help="Model path")
    args = parser.parse_args()

    encode(args.model, 'data')


if __name__ == '__main__':
    main()
