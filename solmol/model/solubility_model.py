import logging

import torch
from torch import nn, Tensor
from torch.nn import Sequential

from solmol.model.modules.pooler import ContextPooler
from solmol.model.utils import get_backbone, get_trainable_parameters


def get_model(name, model, use_lora=False, lora_rank: int = 16, lora_embeddings: bool = True) -> nn.Module:
    backbone = get_backbone(name, use_lora, lora_rank, lora_embeddings)
    models = {
        "esm_rep": EsmSolubilityRep,
        "esm_logits": EsmSolubilityLogits,
    }
    sol_model = models[model](backbone=backbone, use_lora=use_lora).cuda()
    logging.info(f"{get_trainable_parameters(sol_model)} trainable parameters")
    return sol_model


class EsmSolubilityRep(nn.Module):
    def __init__(self, backbone: nn.Module, use_lora: bool = False) -> None:
        super(EsmSolubilityRep, self).__init__()
        self.backbone = backbone
        self.num_layers = self.backbone.num_layers
        self.head = self.get_head(self.backbone.embed_dim, use_lora)

    def get_head(self, embed_dim, use_lora: bool = False) -> nn.Module:
        return Sequential(ContextPooler(embed_dim, 0.1),
                          nn.Linear(embed_dim, 1))

    def get_embeddings(self, x: Tensor) -> Tensor:
        esm_embeddings = self.backbone(x, repr_layers=[self.num_layers])
        embeddings = esm_embeddings['representations'][self.num_layers]
        return embeddings

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.get_embeddings(x)
        x = self.head(embeddings).squeeze()
        output = x

        return output


class EsmSolubilityLogits(EsmSolubilityRep):
    def get_head(self, embed_dim, use_lora: bool = False) -> nn.Module:
        return Sequential(ContextPooler(33, 0.1),
                          nn.Linear(33, 1))

    def get_embeddings(self, x: Tensor) -> Tensor:
        esm_embeddings = self.backbone(x)
        embeddings = esm_embeddings['logits']
        return embeddings
