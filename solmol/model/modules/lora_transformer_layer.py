from esm.modules import ESM1bLayerNorm, ESM1LayerNorm, gelu
from torch import nn

from solmol.model.modules.lora_multihead_attention import LoraMultiheadAttention
import loralib as lora


class LoraTransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv=True,
            use_esm1b_layer_norm=False,
            use_rotary_embeddings: bool = False,
            lora_rank=16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.lora_rank = lora_rank
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = LoraMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
            lora_rank=self.lora_rank
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = lora.Linear(self.embed_dim, self.ffn_embed_dim, r=self.lora_rank)
        self.fc2 = lora.Linear(self.ffn_embed_dim, self.embed_dim, r=self.lora_rank)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
            self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn
