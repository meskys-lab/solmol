import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
import loralib as lora

from solmol.data.dataloader import get_dataloaders
from solmol.learner import get_learner
from solmol.model.solubility_model import get_model
from solmol.model.utils import freeze_backbone, unfreeze, unfreeze_head


def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=None, help='Path to a train csv file')
    parser.add_argument('--val_csv', type=str, default=None, help='Path to a val csv file')
    parser.add_argument('--model_dir', type=str, default="models", help='Path to folder with trained model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--backbone', type=str, default='esm2_t6_8M_UR50D', help='Type of the backbone')
    parser.add_argument('--model', type=str, default='esm_rep', help='Type of the model')
    parser.add_argument('--use_lora', type=bool, default=False, help='Whether to use lora')
    parser.add_argument('--lora_rank', type=int, default=64, help='Rank for lora training')
    parser.add_argument('--esm_model_hub', type=str, default='/mnt/shared/models/esm/weights/hub',
                        help='Path where ESM models are downloaded')

    parser.add_argument('--initial_epochs', type=int, default=3, help='Number of epochs to train head only')
    parser.add_argument('--one_cycle_epochs', type=int, default=12,
                        help='Number of epochs to train one cycle')
    parser.add_argument('--n_one_cycles', type=int, default=3, help='Number of one cycle training')

    args = parser.parse_args()
    return args


def train(args):
    logging.info(f"Starting training with these parameters: {args}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    dataloaders = get_dataloaders(train_data=train_df, val_data=val_df, batch_size=args.batch_size)

    sol_model = get_model(args.backbone, args.model, use_lora=args.use_lora, lora_rank=args.lora_rank)

    learner = get_learner(model=sol_model, dataloaders=dataloaders)

    model_name = get_model_name(sol_model)

    save_config(learner, model_name)

    training_schedule(learner, args)

    path = learner.save(model_name)

    logging.info(f"Model has been save: {path}")


def get_model_name(model: nn.Module) -> str:
    name = model.__class__.__name__
    model_name = f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    return model_name


def save_config(learner, name: str) -> None:
    config_file = (Path(learner.model_dir) / name).with_suffix(".yaml")
    with open(config_file, 'w') as file:
        yaml.dump(vars(args), file)


def training_schedule(learner, args) -> None:
    freeze_backbone(learner.model)
    lr = learner.lr_find().valley
    logging.info(f"For head training {lr} will be used as learning rate ")
    learner.fit(args.initial_epochs, lr=lr, reset_opt=True)

    unfreeze(learner.model)

    if args.use_lora:
        lora.mark_only_lora_as_trainable(learner.model)
        unfreeze_head(learner.model)

    lr_init = learner.lr_find().valley
    for i in range(args.n_one_cycles):
        lr = lr_init / (2 ** i)
        logging.info(f"For training {i} cycle, {lr} will be used as learning rate")
        learner.fit_one_cycle(args.one_cycle_epochs, lr_max=lr, reset_opt=True)

    logging.info("Training has been completed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_train_args()

    torch.hub.set_dir(args.esm_model_hub)

    train(args)
