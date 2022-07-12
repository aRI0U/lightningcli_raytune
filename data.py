from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, data_dir, **dl_kwargs):
        super(MNISTDataModule, self).__init__()

        self.data_dir = data_dir
        self.dl_kwargs = dl_kwargs

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # initialized in setup()
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, **self.dl_kwargs, shuffle=False)
