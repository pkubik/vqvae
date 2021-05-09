import numpy as np
import torch
from torchvision.utils import make_grid

from utils import load_cifar
from utils import load_vqvae
from models.vqvae import VQVAE
import matplotlib.pyplot as plt


def display_image_grid(x):
    x = make_grid(x.cpu().detach()+0.5)
    x = x.numpy()
    fig = plt.imshow(np.transpose(x, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def reconstruct(model: VQVAE, images: list, device, plot_path: str = None):
    x = torch.stack(images)
    x = x.to(device)
    _, hx, _ = model(x)

    concated_x = torch.cat((x, hx))
    display_image_grid(concated_x)
    if plot_path:
        plt.savefig(plot_path)
    else:
        plt.show()


def uniform_sample(model: VQVAE, num_samples: int, device, plot_path: str = None):
    code_shape = model.encode(torch.zeros((num_samples, 3, 32, 32), device=device)).shape
    print('Latent code shape:', code_shape)
    code = torch.randint(0, model.vector_quantization.embedding.num_embeddings, code_shape, device=device)
    emb = model.vector_quantization.embedding(code).permute(0, 3, 1, 2)
    hx = model.decoder(emb)

    display_image_grid(hx)
    if plot_path:
        plt.savefig(plot_path)
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-r', '--reconstruction', action='store_true')
    parser.add_argument('-u', '--uniform-sample', action='store_true')
    parser.add_argument('-g', '--gpu', action='store_true', default=None)
    parser.add_argument('--no-gpu', action='store_false', default=None)
    parser.add_argument('-n', '--num-samples', type=int, default=16)
    args = parser.parse_args()

    use_gpu = args.gpu
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    vqvae, data = load_vqvae(args.model, device)
    params = data["hyperparameters"]

    print(f"Loaded model {args.model}")
    data = None

    if args.reconstruction:
        if data is None:
            _, data = load_cifar()
        reconstruct(vqvae, [data[i][0] for i in range(args.num_samples)], device)

    if args.uniform_sample:
        uniform_sample(vqvae, args.num_samples, device)


if __name__ == '__main__':
    main()
