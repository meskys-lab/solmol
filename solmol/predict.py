import argparse
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import yaml
from esm import FastaBatchedDataset, BatchConverter
from fastai.learner import load_model
from torch import nn
from torch.utils.data import DataLoader

from solmol.data.dataloader import ALPHABET

from solmol.model.solubility_model import get_model


def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to a fasta file for predictions')
    parser.add_argument('--esm_model_hub', type=str, default='/mnt/shared/models/esm/weights/hub',
                        help='Path where ESM models are downloaded')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output', type=str, default="example/predictions.csv",
                        help='Full path where to store predictions')
    args = parser.parse_args()
    return args


def predict(args: argparse.Namespace):
    logging.info(f"Starting predicting using model {args.model}")

    with open(Path(args.model).with_suffix(".yaml")) as f:
        config = yaml.safe_load(f)

    pred_dataset = FastaBatchedDataset.from_file(args.fasta_path)

    collate_fn = BatchConverter(ALPHABET)
    pred_dataloader = DataLoader(pred_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    sol_model = get_model(config["backbone"], config["model"], use_lora=config["use_lora"],
                               lora_rank=config["lora_rank"])
    load_model(args.model, sol_model, None, with_opt=False)
    sol_model.eval()

    results = make_predictions(sol_model, pred_dataloader)
    save_output(args, results)


def save_output(args: argparse.Namespace, results: pd.DataFrame) -> None:
    results_df = pd.DataFrame(results)
    logging.info(f"Prediction will be save in {args.output}")
    results_df.to_csv(args.output, index=None)


def make_predictions(model: nn.Module, pred_dataloader: DataLoader) -> dict:
    results = {"ids": [], "sequence": [], "prediction": []}
    with torch.no_grad():
        for b in pred_dataloader:
            output = model(b[2].to('cuda'))
            output = output.detach().cpu().numpy()
            output = [output] if output.shape == () else output
            results["ids"].extend(b[0])
            results["sequence"].extend(b[1])
            results["prediction"].extend(output)
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_train_args()
    torch.hub.set_dir(args.esm_model_hub)

    predict(args)
