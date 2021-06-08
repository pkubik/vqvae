import argparse
from pathlib import Path
from typing import Union
import torch
import numpy as np

from utils import load_data_and_data_loaders
from utils import load_vqvae


def encode_from_loader(vqvae, loader, output_path: Path, device):
    encodings_batches = []
    labels_batches = []
    for (x, y) in loader:
        x = x.to(device)
        encodings = vqvae.encode(x)
        encodings_batches.append(encodings)
        labels_batches.append(y)
    all_encoding = torch.cat(encodings_batches, 0).cpu().numpy()
    all_labels = torch.cat(labels_batches)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(output_path, x=all_encoding, y=all_labels)


def encode(model_path: Union[str, Path], output_path: Union[str, Path]):
    model_path = Path(model_path)
    output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae, config = load_vqvae(model_path, device)
    params = config["hyperparameters"]
    dataset = params['dataset']

    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders(dataset, 128)

    print(f"Encoding {dataset} test...")
    encode_from_loader(vqvae, validation_loader, output_path / 'test' / f'encoded_{dataset}.npz', device)

    print(f"Encoding {dataset} train...")
    encode_from_loader(vqvae, training_loader, output_path / 'train' / f'encoded_{dataset}.npz', device)


def main():
    parser = argparse.ArgumentParser("Encode whole dataset using selected model.")
    parser.add_argument("model", help="Model path")
    args = parser.parse_args()

    encode(args.model, 'data')


if __name__ == '__main__':
    main()
