import torch as th
import torch.nn as nn
import torch.nn.functional as F

from queue import Queue
from .configuration_wpt import WeightedPerceptualTransferConfig
from ..layers import *
from ..attention import MultiHeadAttention
from typing import Any
from torchdiffeq import odeint_adjoint as odeint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput




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
                values = self._ode_forward(z0[bidx, None], 
                                        times, 
                                        chunk_size=chunk_size)
                stack.append(values)
            return th.cat(stack, dim=0)
        else:
            return self._ode_forward(z0, t, chunk_size=chunk_size)

class TransferInterpolationBlock(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(TransferInterpolationBlock, self).__init__()
        tweights = th.zeros((1, config.lfeatures, config.tinterpolation_size, 1))
        self.register_buffer("time_weights", tweights)

    def forward(self, t: TensorType["B", "T"]):
        if t.ndim == 1 or t.shape[0] == 1:
            T = t.squeeze().size(0)
            tgrid = t.view(1, 1, T, 1)
            tgrid = th.cat([tgrid, th.zeros_like(tgrid)], dim=-1)
            return F.grid_sample(self.time_weights, tgrid).squeeze().transpose(0, 1)
        elif t.ndim == 2:
            T = t.size(-1)
            tgrids = []
            for tgrid in t:
                tgrid = tgrid.view(1, 1, T, 1)
                tgrid = th.cat([tgrid, th.zeros_like(tgrid)], dim=-1)
                tvalues = F.grid_sample(self.time_weights, tgrid).squeeze().transpose(0, 1)
                tgrids.append(tvalues)
            return th.stack(tgrids, dim=0)

class TimeTransferFusion(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(TimeTransferFusion, self).__init__()
        self.gate1 = nn.Sequential(nn.Linear(config.lfeatures, config.lfeatures),
                                FiltrationBlock(config.lfeatures))
        self.gate2 = nn.Sequential(nn.Linear(config.lfeatures, config.lfeatures),
                                FiltrationBlock(config.lfeatures))
        self.aligner = nn.Linear(config.lfeatures*2, config.lfeatures)

    def forward(self, zt_flow: TensorType["B", "T", "C"],
                zt_descrite: TensorType["B", "T", "C"]):
        ztf = self.gate1(zt_flow)
        ztd = self.gate2(zt_descrite)
        print(ztf.shape, ztd.shape)
        zt_comb = th.cat([ztf, ztd], dim=-1)
        return self.aligner(zt_comb)

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
        return {"temporal_cls_tokens": tcls_tokens,
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
        self.tt_fuse = TimeTransferFusion(config)
        self.tt_descrite = TransferInterpolationBlock(config)

    def forward(self, patch_tokens: th.Tensor,
                cls_token: th.Tensor,
                timestampts: th.Tensor,
                tmask: Optional[th.BoolTensor | th.LongTensor]=None):

        B = cls_token.size(0)
        z0 = self.encoder(cls_token).squeeze()
        zt_flow = self.flow_model(z0, timestampts, self.cfg.time_chunk_size)
        zt_descrite = self.tt_descrite(timestampts)
        if zt_descrite.ndim == 2:
            zt_descrite = zt_descrite[None, ...].repeat(B, 1, 1)
        zt = self.tt_fuse(zt_flow, zt_descrite)
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

    def forward(self, image: TensorType["B", "C", "W", "H"],
                get_intermediates: bool=False):
        embeddings = self.patch_embed(image)
        tokens = th.cat([embeddings, self.cls.view(1, 1, -1).repeat(10, 1, 1)], dim=1)
        intermediates: Dict[str, th.Tensor | Tuple[th.Tensor]] = {} if get_intermediates else None
        for idx, block in enumerate(self.blocks):
            bout = block(tokens, use_cache=False)
            tokens = bout.last_features
            if intermediates is not None:
                intermediates[idx] = bout.hidden_features

        return {"patch_tokens": tokens[:, :-1, :],
                "cls_token": tokens[:, -1, :],
                "intermediates": intermediates}

#============================ViT Modelling Part=============================================================



@dataclass 
class WPTOutput:
    patch_tokens:          Optional[th.Tensor]=None
    cls_token:             Optional[th.Tensor]=None
    intermediates:         Optional[th.Tensor]=None
    temporal_cls_tokens:   Optional[th.Tensor]=None
    channels_output:       Optional[th.Tensor]=None
    

class TemporalTrnasformerDequeue:
    def __init__(self, size: int):
        self.size = size
        self.dequeue = None

class WeightedPerceptualTransferModel(PreTrainedModel):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(WeightedPerceptualTransferModel, self).__init__(config)
        self.visual = VisualTransformer(config)
        self.ptrnasnet = PerceptualTransferModel(config)

    def forward(self, image: TensorType["B", "W", "H", "C"],
                t: Optional[TensorType["B", "Time"]]=None,
                get_vit_intermediates: bool=False):
        output = self.visual(image, get_vit_intermediates)
        if t is not None:
            cls_token = output["cls_token"][:, None, :]
            temporal_out = self.ptrnasnet(output["patch_tokens"], 
                                cls_token,
                                timestampts=t)
            output.update(temporal_out)
        return WPTOutput(**output)
    
if __name__ == "__main__":
    config = WeightedPerceptualTransferConfig(in_channels=3, 
                                            out_channels=13,
                                            visual_features=556,
                                            image_size=448,
                                            time_chunk_size=54)
    # tgrid = TransferInterpolationBlock(config)
    # times = th.normal(0, 1, (10, 100))
    # tfeatures = tgrid(times)
    # print(tfeatures.shape)
    model = WeightedPerceptualTransferModel(config)
    print(f"Ntotal: {sum([p.numel() for p in model.parameters()])}")
    data = th.normal(0, 1, (10, 3, 448, 448))
    times = th.linspace(0, 1, 1000)
    times = times[None, :].repeat(10, 1)
    out = model(data, times)
    print(out.patch_tokens.shape,
        out.temporal_cls_tokens.shape,
        out.channels_output.shape)
    
    # print(fout.fused_features.shape, fout.signal_values.shape)
    
