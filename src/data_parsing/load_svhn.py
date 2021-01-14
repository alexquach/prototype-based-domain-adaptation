from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision.transforms import Grayscale, Compose, CenterCrop, ToTensor

def load_svhn_dataloader(bs=64, greyscale=False):
    if greyscale:
        transform = Compose([
            CenterCrop(28),
            Grayscale(),
            ToTensor()
        ])
    else:
        transform = Compose([
            CenterCrop(28),
            ToTensor()
        ])

    train_ds = SVHN("data/svhn", split='train', transform=transform, download=True)
    test_ds = SVHN("data/svhn", split='test', transform=transform, download=True)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs * 2)

    return train_dl, test_dl