import numpy as np
import os
import torch
from torchvision.utils import make_grid

from pixelcnn.net import PixelCNN
from utils import load_cifar, load_data_and_data_loaders
from utils import load_mnist
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
    plt.title('Reconstruct')
    display_image_grid(concated_x)
    if plot_path:
        plt.savefig(plot_path)
    else:
        plt.show()


def uniform_sample(model: VQVAE, num_samples: int, device, plot_path: str = None):
    code_shape = model.encode(torch.zeros((num_samples, 3, 32, 32), device=device)).shape
    print('Latent code shape:', code_shape)
    plt.title('Uniform sample')
    code = torch.randint(0, model.vector_quantization.embedding.num_embeddings, code_shape, device=device)
    decode(model, code, plot_path)


def decode(model: VQVAE, code, plot_path: str = None):
    emb = model.vector_quantization.embedding(code.squeeze(1)).permute(0, 3, 1, 2)
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
    parser.add_argument('-p', '--pixelcnn', type=str, default=None)
    parser.add_argument('-g', '--gpu', action='store_true', default=None)
    parser.add_argument('--no-gpu', action='store_false', default=None)
    parser.add_argument('-n', '--num-samples', type=int, default=16)
    parser.add_argument('--plot_path', type=str, default=None)
    args = parser.parse_args()

    recon_path = None
    sample_path = None
    pixelcnn_sample_path = None
    if args.plot_path:
        os.makedirs(args.plot_path, exist_ok=True)
        
        recon_path = os.path.join(args.plot_path, 'recon.png')
        sample_path = os.path.join(args.plot_path, 'sample.png')
        pixelcnn_sample_path = os.path.join(args.plot_path, 'pixelcnn_sample.png')

    use_gpu = args.gpu
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    vqvae, config = load_vqvae(args.model, device)
    params = config["hyperparameters"]

    print(f"Loaded model {args.model}")
    data = None

    if args.reconstruction:
        if data is None:
            _, data, _, _, _ = load_data_and_data_loaders(params['dataset'], params['batch_size'])
        reconstruct(vqvae, [data[i][0] for i in range(args.num_samples)], device, plot_path=recon_path)

    if args.uniform_sample:
        uniform_sample(vqvae, args.num_samples, device, plot_path=sample_path)

    if args.pixelcnn:
        ckpt = torch.load(args.pixelcnn)
        pixelcnn_state = ckpt['model']
        cfg = ckpt['config']
        pixelcnn = PixelCNN(cfg).to(device)
        pixelcnn.load_state_dict(pixelcnn_state)
        code_shape = vqvae.encode(torch.zeros((1, 3, 32, 32), device=device)).shape
        code = pixelcnn.sample(code_shape, args.num_samples, device=device)
        plt.title('PixelCNN decode')
        decode(vqvae, code, plot_path=pixelcnn_sample_path)


if __name__ == '__main__':
    main()
