import cv2
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    Creates block dataset of images with 3 channels
    if crop is True, crops to square
    if resize is True, resizes to 32X32
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None, resize=True, crop=False):
        print('Loading block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')

        if crop:
            width = data.shape[1]
            height = data.shape[2]
            new_size = min(width, height)

            offset_w = (width - new_size) // 2
            offset_h = (height - new_size) // 2

            data = data[:,offset_w:(offset_w + new_size),offset_h:(offset_h + new_size),:]

        if resize:
            data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
                32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset 
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading latent block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading latent block data')
        
        self.data = data[:-500] if train else data[-500:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)
