import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .configuration_wpt import WeightedPerceptualTransferConfig
from ..layers import *
from ..attention import MultiHeadAttention
from typing import Any
from torchdiffeq import odeint_adjoint as odeint
from transformers import PreTrainedModel




#============================Perceptual Fusion Modeling Part=============================================================

class FlowFunction(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(FlowFunction, self).__init__()
        self.projection = Mlp(features=config.lfeatures, 
                            activation=config.lact_fn)
        self.norm = FiltrationBlock(config.lfeatures)

    def forward(self, t: Any, xt: th.Tensor):
        xt = self.projection(xt)
        xt = xt if not hasattr(self, "norm") else self.norm(xt)
        return xt

class FlowModel(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(FlowModel, self).__init__()
        self.fn = FlowFunction(config)

    def _ode_forward(self, z0: th.Tensor, 
                    times: th.Tensor,
                    chunk_size: Optional[int]=None):
        if (chunk_size is None) or (chunk_size >= times.shape[0]):
            return odeint(self.fn, z0, times)
        else:
            zt0 = z0.clone()
            n_chunks = int(times.shape[0] // chunk_size)
            result = []
            for idx in range(n_chunks):
                chunk_times = times[idx*chunk_size: (idx + 1)*chunk_size]
                values = odeint(self.fn, zt0, chunk_times)
                zt0 = values[-1, ...]
                result.append(values)
            if (times.shape[0] % chunk_size) != 0:
                last_times = times[n_chunks*chunk_size:]
                values = odeint(self.fn, zt0, last_times)
                result.append(values)
            result = th.cat(result, dim=0).transpose(0, 1)
            return result
                
    def forward(self, z0: TensorType["B", "C"], 
                t: TensorType["B", "T"],
                chunk_size: Optional[int]=None):
        if t.ndim == 2:
            stack = []
            assert (t.size(0) == z0.size(0))
            B = z0.size(0)
            for bidx in range(B):
                times = t[bidx]
                # values = odeint(self.fn, z0[bidx], times)
                values = self._ode_forward(z0, 
                                        times, 
                                        chunk_size=chunk_size)
                stack.append(values)
            return th.stack(stack, dim=0)
        else:
            # return odeint(self.fn, z0, t).transpose(0, 1)
            return self._ode_forward(z0, t, chunk_size=chunk_size)

class Encoder(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(Encoder, self).__init__()
        self.projection = Mlp(features=config.vfeatures,
                            out_features=config.lfeatures,
                            activation=config.lact_fn)
        self.norm = FiltrationBlock(config.lfeatures)
        self.attn = MultiHeadAttention(features=config.lfeatures,
                                    nheads=config.attention_heads,
                                    scoring_fn=config.attention_scoring,
                                    heads_reduction=config.attention_reduction)

    def forward(self, x: TensorType["B", "S", "C"]):
        x = self.projection(x)
        x = self.norm(x)
        x, _ = self.attn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(Decoder, self).__init__()
        self.projection = Mlp(features=config.lfeatures,
                                    out_features=config.vfeatures,
                                    activation=config.lact_fn)
        self.norm = FiltrationBlock(config.vfeatures)

    def forward(self, x: th.Tensor):
        x = self.projection(x)
        x = self.norm(x)
        return x

@dataclass
class FuseSignalOutput:
    fused_features: Optional[th.Tensor]=None
    signal_values: Optional[th.Tensor]=None

class FuseChannelsHead(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(FuseChannelsHead, self).__init__()
        pwn = int(config.image_size[0] // config.patch_size[0])
        phn = int(config.image_size[1] // config.patch_size[1])
        pn = pwn * phn
        self.contribg_gating = Mlp(config.vfeatures, pn, 0.0, "sigmoid")
        self.projection = Mlp(config.vfeatures, config.out_channels, 0.0, "relu")
        self.norm = FiltrationBlock(config.out_channels)
        self.register_buffer("channels_gates", th.zeros((config.out_channels)))


    def _peak_estimation(self, svalues: TensorType["B", "T", "S", "C"]):
        svalues = svalues * self.channels_gates.view(1, 1, 1, -1)
        svalues = svalues.max(dim=-2).values
        return svalues

    def _chunk_forward(self, patch_tokens: TensorType["B", "S", "C"],
                tcls_tokens: TensorType["B", "T", "C"],
                tmask: Optional[TensorType["B", "T"]]=None,
                chunk_size: Optional[int]=None):
        if (chunk_size is None) or (chunk_size >= tcls_tokens.shape[1]):
            return self._forward_impl(patch_tokens, tcls_tokens, tmask)
        else:
            (_, T, _) = tcls_tokens.shape
            n_chunks = int(T // chunk_size)
            result: Dict[str, List[th.Tensor]] = {}
            def update_results(chunk_tcls: th.Tensor):
                foutput = self._forward_impl(patch_tokens, chunk_tcls, tmask)
                for (k, v) in foutput.items():
                    if k not in result: result[k] = list([v, ])
                    else: result[k].append(v)

            for idx in range(n_chunks):
                chunk_tcls = tcls_tokens[:, idx*chunk_size: (idx + 1)*chunk_size, :]
                update_results(chunk_tcls=chunk_tcls)
            if (T % chunk_size) != 0:
                chunk_tcls = tcls_tokens[:, n_chunks*chunk_size:, :]
                update_results(chunk_tcls=chunk_tcls)

            for (k, v) in result.items():
                result[k] = th.cat(v, dim=1)

            return result
        
    def _forward_impl(self, patch_tokens: TensorType["B", "S", "C"],
                tcls_tokens: TensorType["B", "T", "C"],
                tmask: Optional[TensorType["B", "T"]]=None):

        (B, N, C) = patch_tokens.shape
        (_, T, _) = tcls_tokens.shape
        tcls_tokens = (tcls_tokens if tmask is None else tcls_tokens[tmask])
        contribs = self.contribg_gating(tcls_tokens)
        tokens = patch_tokens.view(B, 1, N, C) * contribs.view(B, T, -1, 1)
        tokens = (tokens * tcls_tokens.view(B, T, 1, -1))
        channels_out = self.projection(tokens)
        channels_out = self._peak_estimation(channels_out)
        return {"temporal_patch_tokens": tokens,
                "temporal_cls_tokens": tcls_tokens,
                "channels_output": channels_out}
    
    def forward(self, patch_tokens: TensorType["B", "S", "C"],
                tcls_tokens: TensorType["B", "T", "C"],
                tmask: Optional[TensorType["B", "T"]]=None,
                chunk_size: Optional[int]=None):
        return self._chunk_forward(patch_tokens,
                                tcls_tokens,
                                tmask,
                                chunk_size)

class PerceptualTransferModel(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(PerceptualTransferModel, self).__init__()
        self.cfg = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.flow_model = FlowModel(config)
        self.fuse_head = FuseChannelsHead(config)

    def forward(self, patch_tokens: th.Tensor,
                cls_token: th.Tensor,
                timestampts: th.Tensor,
                tmask: Optional[th.BoolTensor | th.LongTensor]=None):

        z0 = self.encoder(cls_token).squeeze()
        zt = self.flow_model(z0, timestampts, self.cfg.time_chunk_size)
        xt = self.decoder(zt)
        return self.fuse_head(patch_tokens, xt, tmask, self.cfg.time_chunk_size)
#============================<(Perceptual Fusion Modeling Part)>=============================================================


#============================ViT Modelling Part=============================================================
class PatchEmbedding(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(PatchEmbedding, self).__init__()
        self.cfg = config
        # print(config.channels, config.vfeatures, type(config.patch_size), config.patch_size)
        self.conv = nn.Conv2d(config.in_channels, 
                            config.vfeatures,
                            config.patch_size,
                            config.patch_size)
        if config.use_adanorm:
            self.norm = FiltrationBlock(config.vfeatures)

    def forward(self, image: TensorType["B", "C", "W", "H"]):
        (B, _, W, H) = image.shape
        assert (image.size(-1) % self.cfg.patch_size[-1] == 0),\
        ("wrong image_size: {image.shape[-2:]}"
        "must be dividble by patch_size: {self.cfg.patch_size}")
        embeddings = self.conv(image)
        embeddings = embeddings\
            .view(B, -1, (W // self.cfg.patch_size[0]) 
                  * (H // self.cfg.patch_size[1]))\
            .transpose(1, 2)
        embeddings = embeddings if not hasattr(self, "norm") else self.norm(embeddings)
        return embeddings

@dataclass 
class ViTOutput:
    patch_tokens: Optional[th.Tensor]=None
    cls_token: Optional[th.Tensor]=None
    intermediates: Optional[Dict[str, Tuple[th.Tensor] | th.Tensor]]=None

class VisualTransformer(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(VisualTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList([
            BlockStack(config.vfeatures,
                    config.block_depth, 0.0,
                    config.vact_fn,
                    config.use_adanorm,
                    config.attention_heads,
                    config.attention_reduction,
                    config.attention_scoring,
                    config.skip)
            for _ in range(config.aggregation_depth)
        ])
        cls_token = th.zeros((config.vfeatures, ))
        self.register_buffer("cls", cls_token)

    def forward(self, image: TensorType["B", "C", "W", "H"]):
        embeddings = self.patch_embed(image)
        tokens = th.cat([embeddings, self.cls.view(1, 1, -1).repeat(10, 1, 1)], dim=1)
        intermediates: Dict[str, th.Tensor | Tuple[th.Tensor]] = {}
        for idx, block in enumerate(self.blocks):
            bout = block(tokens, use_cache=False)
            tokens = bout.last_features
            intermediates[idx] = bout.hidden_features

        return {"patch_tokens": tokens[:, :-1, :],
                "cls_token": tokens[:, -1, :],
                "intermediates": intermediates}

#============================ViT Modelling Part=============================================================



@dataclass 
class WPTOutput:
    patch_tokens: Optional[th.Tensor]=None
    cls_token: Optional[th.Tensor]=None
    intermediates: Optional[th.Tensor]=None
    temporal_patch_tokens: Optional[th.Tensor]=None
    temporal_cls_tokens: Optional[th.Tensor]=None
    channels_output: Optional[th.Tensor]=None
    
    
class WeightedPerceptualTransferModel(PreTrainedModel):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(WeightedPerceptualTransferModel, self).__init__(config)
        self.visual = VisualTransformer(config)
        self.ptrnasnet = PerceptualTransferModel(config)

    def forward(self, image: TensorType["B", "W", "H", "C"],
                t: Optional[TensorType["B", "Time"]]=None):
        output = self.visual(image)
        if t is not None:
            temporal_out = self.ptrnasnet(output["patch_tokens"], 
                                output["cls_token"][:, None, :],
                                timestampts=t)
            output.update(temporal_out)
        return WPTOutput(**output)
    

    
    

        
