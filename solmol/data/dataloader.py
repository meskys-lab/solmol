from typing import Callable

import numpy as np
import pandas as pd
import torch
from esm import FastaBatchedDataset, BatchConverter, Alphabet
from fastai.data.core import DataLoaders
from torch.utils.data import DataLoader

ALPHABET = Alphabet.from_architecture("ESM-1b")


def get_train_dataset(data: pd.DataFrame) -> FastaBatchedDataset:
    return FastaBatchedDataset(sequence_labels=data.solubility, sequence_strs=data.sequence)


def get_dataloader(collate_fn: Callable, data: pd.DataFrame, batch_size=8, label_smoothing=False) -> DataLoader:
    dataset = get_train_dataset(data)

    def reorder(items):
        items = collate_fn(items)
        label = torch.tensor(np.asarray(items[0]), dtype=torch.float32).to('cuda')
        if label_smoothing:
            label = label + ((1.0 - (2 * label)) * (torch.rand_like(label) / 10.0))
        return items[2].to('cuda'), label

    return DataLoader(dataset, collate_fn=reorder, batch_size=batch_size, shuffle=True)


def get_dataloaders(train_data: pd.DataFrame, val_data: pd.DataFrame, batch_size=8) -> DataLoaders:
    collate_fn = BatchConverter(ALPHABET)

    train_dataloader = get_dataloader(collate_fn, train_data, batch_size, label_smoothing=True)
    val_dataloader = get_dataloader(collate_fn, val_data, batch_size, label_smoothing=False)

    return DataLoaders(train_dataloader, val_dataloader, device='cuda')
