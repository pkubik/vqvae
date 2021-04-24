import logging
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

from pixelcnn.net import GatedPixelCNN

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)


@dataclass
class TrainConfig:
    max_epoch: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    log_interval: int = 10
    dataset: str = 'mnist'
    device: str = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NetConfig:
    num_image_channels: int = 1
    image_size: int = 28
    num_base_channels = 64
    num_levels: int = 512
    num_layers: int = 5


class Trainer:
    def __init__(self, train_config: TrainConfig, net_config: NetConfig):
        self.train_config = train_config
        self.net_config = net_config
        self.device = torch.device(train_config.device)
        self.net = GatedPixelCNN(
            net_config.num_levels,
            net_config.num_base_channels,
            net_config.num_layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=train_config.learning_rate)
        self.dataset = datasets.MNIST(
            'data', train=True, download=True,
            transform=transforms.ToTensor())

    def train_single_epoch(self):
        data_loader_kwargs = {'batch_size': self.train_config.batch_size}
        if self.train_config.device == 'cuda':
            data_loader_kwargs.update({
                'num_workers': 2,
                'pin_memory': True,
                'shuffle': True
            })

        data_loader = data.DataLoader(self.dataset, **data_loader_kwargs)

        losses = []
        for batch_idx, (x, y) in enumerate(data_loader):
            x = (x[:, 0] / 255. * (self.net_config.num_levels - 1)).long().to(self.device)
            y = y.to(self.device)

            logits = self.net(x, y)

            flattened_logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, self.net_config.num_levels)
            loss = nn.CrossEntropyLoss()(
                flattened_logits,
                x.view(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            if len(losses) >= self.train_config.log_interval:
                mean_loss = np.mean(losses)
                logging.info(f'After {batch_idx + 1} steps, loss: {mean_loss}')
                losses = []

    def train(self):
        for epoch in range(self.train_config.max_epoch + 1):
            logging.info(f"Starting epoch {epoch}")
            self.train_single_epoch()


if __name__ == '__main__':
    trainer = Trainer(TrainConfig(), NetConfig())
    trainer.train()
