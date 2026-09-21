import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from ...layers import get_activation, Transformer
from typing import Optional 
from .reward_configuring import RewardConfig


class RewardModel(PreTrainedModel):
    config_class = RewardConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        cs, d = config.chunk_size, config.chunk_dim
        h = config.backbone_hidden_size

        self.backbone = AutoModel.from_config(config.backbone_config)

        self.chunk_proj = nn.Sequential(
            nn.Linear(h, d),
            nn.LayerNorm(d),
            get_activation(config.activation),
        )
        self.aggregation = Transformer(cs*d,
                                        config.aggregation_depth,
                                        config.aggregation_heads,
                                        config.aggregation_n_attn_heads,
                                        config.dropout,
                                        config.activation,
                                        residual_connections=config.aggregation_residuals)
        self.chunk_score = nn.Linear(cs * d, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(cs * d, config.num_outputs)
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone_name_or_path: str,
        backbone_kwargs: Optional[dict] = None,
        **config_kwargs,
    ) -> "RewardModel":
        backbone = AutoModel.from_pretrained(backbone_name_or_path, **(backbone_kwargs or {}))
        config = RewardConfig(backbone_config=backbone.config, **config_kwargs)
        model = cls(config)
        model.backbone = backbone
        return model

    def forward(
        self,
        input_ids: Optional[th.LongTensor] = None,
        attention_mask: Optional[th.Tensor] = None,
        labels: Optional[th.Tensor] = None,
        **backbone_kwargs,
    ) -> SequenceClassifierOutput:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **backbone_kwargs
        ).last_hidden_state
        B, L, _ = hidden.shape
        cs, d = self.config.chunk_size, self.config.chunk_dim

        # if attention_mask is None:
        #     attention_mask = th.ones(B, L, dtype=th.long, device=hidden.device)
        # mask = attention_mask.long()

        pad = (-L) % cs
        if pad:
            hidden = F.pad(hidden, (0, 0, 0, pad))
            # mask = F.pad(mask, (0, pad), value=0)
        # mask = mask.bool()
        n = (L + pad) // cs

        tokens = self.chunk_proj(hidden)
        chunks = tokens.reshape(B, n, cs * d)
        chunks = self.aggregation(chunks, attn_mask=~chunk_mask)

        m = chunk_mask.unsqueeze(-1).to(chunks.dtype)
        red = self.config.chunk_reduction
        if red == "weighted":
            scores = self.chunk_score(chunks).squeeze(-1)
            scores = scores.masked_fill(~chunk_mask, th.finfo(scores.dtype).min)
            weights = scores.softmax(dim=1).unsqueeze(-1)
            pooled = (chunks * weights).sum(dim=1)
        elif red == "mean":
            pooled = (chunks * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        elif red == "sum":
            pooled = (chunks * m).sum(dim=1)
        else:
            raise ValueError(f"unknown chunk_reduction: {red}")

        logits = self.head(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.mse_loss(logits.squeeze(-1), labels.to(logits.dtype))
        return SequenceClassifierOutput(loss=loss, 
                                        logits=logits,
                                        last)