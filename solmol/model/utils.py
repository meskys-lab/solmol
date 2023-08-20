import logging
import re

import torch
from esm.model.esm2 import ESM2
from torch import nn

from solmol.data.dataloader import ALPHABET
from solmol.model.lora_model import LoRAESM2


def get_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if n.startswith("backbone"):
            p.requires_grad = False

    print(f"Model has {get_trainable_parameters(model)} parameters")


def freeze_embeddings(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'backbone.embed_tokens' in n:
            p.requires_grad = False

    print(f"Model has {get_trainable_parameters(model)} parameters")


def unfreeze(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = True

    print(f"Model has {get_trainable_parameters(model)} parameters")


def unfreeze_head(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if n.startswith("head"):
            p.requires_grad = True

    print(f"Model has {get_trainable_parameters(model)} parameters")


def upgrade_state_dict(state_dict: dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def get_backbone(name: str, use_lora: bool, lora_rank: int = 16, lora_embeddings: bool = True) -> nn.Module:
    path = f"{torch.hub.get_dir()}/checkpoints/{name}.pt"

    logging.info(f"Loading model from {path}")

    model_data = torch.load(path, map_location="cpu")
    cfg = model_data["cfg"]["model"]

    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)

    if use_lora:
        model = LoRAESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            alphabet=ALPHABET,
            token_dropout=cfg.token_dropout,
            lora_rank=lora_rank,
            lora_embeddings=lora_embeddings
        )
    else:
        model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            alphabet=ALPHABET,
            token_dropout=cfg.token_dropout
        )

    model.load_state_dict(state_dict, strict=False)

    trainable_params = get_trainable_parameters(model)

    logging.info(f"Model {name} loaded. It has {trainable_params} trainable parameters")

    return model
