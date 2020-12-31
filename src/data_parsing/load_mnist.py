from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = Path('data')
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

def load_mnist_nparray():
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        return x_train, y_train, x_valid, y_valid

def load_mnist_tensors():
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, load_mnist_nparray()
    )

    return x_train, y_train, x_valid, y_valid

def load_mnist_dataloader(bs=64):
    x_train, y_train, x_valid, y_valid = load_mnist_tensors()

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    return train_dl, valid_dl