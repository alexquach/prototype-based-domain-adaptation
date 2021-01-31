from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import SVHN
from torchvision.transforms import Grayscale, Compose, CenterCrop, ToTensor

def load_svhn_dataloader(bs=64, greyscale=False, training_fraction=None):
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

    if training_fraction:
        last_idx = (int) (training_fraction * len(train_ds))
        train_dl = DataLoader(train_ds, batch_size=bs, sampler=SubsetRandomSampler(range(last_idx)))
    else:
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs * 2)

    return train_dl, test_dl