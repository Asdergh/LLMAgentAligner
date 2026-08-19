import torch as th
import torch.nn as nn
import torch.nn.functional as F

from queue import Queue
from .configuration_wpt import WeightedPerceptualTransferConfig
from ..layers import *
from ..attention import MultiHeadAttention
from typing import Any, Literal
from torchdiffeq import odeint_adjoint as odeint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from tqdm import tqdm




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
            return odeint(self.fn, z0, times).transpose(0, 1)
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
        if t.ndim == 0 or (t.ndim == 1 or t.shape[0] == 1):
            T = (t.squeeze().size(0) if t.ndim != 0 else 1)
            tgrid = t.view(1, 1, T, 1)
            tgrid = th.cat([tgrid, th.zeros_like(tgrid)], dim=-1)
            sampled = F.grid_sample(self.time_weights, tgrid).squeeze()
            if sampled.ndim != 1:
                sampled = sampled.transpose(0, 1)
            return sampled
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

class FuseChannelsHead(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(FuseChannelsHead, self).__init__()
        self.config = config
        pwn = int(config.image_size[0] // config.patch_size[0])
        phn = int(config.image_size[1] // config.patch_size[1])
        pn = pwn * phn
        self.channels_gating = Mlp(config.vfeatures, pn, 0.0, "sigmoid")
        if config.n_classes is not None:
            self.logits_gating = Mlp(config.vfeatures, pn, 0.0, "sigmoid")
            self.logits_projection = Mlp(config.vfeatures, config.n_classes, 0.0, "sigmoid")
        self.channels_projection = Mlp(config.vfeatures, config.out_channels, 0.0, "relu")
        self.norm = FiltrationBlock(config.out_channels)
        self.register_buffer("channels_gates", th.zeros((config.out_channels, )))
        if config.n_classes is not None:
            self.register_buffer("logits_gates", th.zeros(config.n_classes, ))

    def _peak_estimation(self, svalues: TensorType["B", "T", "S", "C"], 
                        modality: Literal["channels", "classes"]):
        assert hasattr(self, f"{modality}_gates")
        svalues = svalues * getattr(self, f"{modality}_gates").view(1, 1, 1, -1)
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

    def _gated_forward_impl(self, patch_tokens: TensorType["B", "S", "C"],
                            tcls_tokens: TensorType["B", "T", "C"],
                            modality: Literal["channels", "classes"]):
        (B, N, C) = patch_tokens.shape
        (_, T, _) = tcls_tokens.shape
        assert hasattr(self, f"{modality}_gating") \
            and hasattr(self, f"{modality}_projection"), \
        (f"wrong chanels modality type: {modality}")
        contribs = getattr(self, f"{modality}_gating")(tcls_tokens)
        tokens = patch_tokens.view(B, 1, N, C) * contribs.view(B, T, -1, 1)
        tokens = (tokens * tcls_tokens.view(B, T, 1, -1))
        output = getattr(self, f"{modality}_projection")(tokens)
        output = self._peak_estimation(output, modality)
        return output
    
    def _forward_impl(self, patch_tokens: TensorType["B", "S", "C"],
                tcls_tokens: TensorType["B", "T", "C"],
                tmask: Optional[TensorType["B", "T"]]=None):

        if (tmask is not None) and (tmask.size(1) == tcls_tokens.size(1)):
            tcls_tokens = tcls_tokens[tmask]
        channels_out = self._gated_forward_impl(patch_tokens, 
                                                tcls_tokens,
                                                "channels")
        logits_out = None
        if (self.config.n_classes is not None) and (self.config.n_classes != 0):
            logits_out = self._gated_forward_impl(patch_tokens,
                                                    tcls_tokens,
                                                    "logits")
        return {"temporal_cls_tokens": tcls_tokens,
                "channels_output": channels_out,
                "logits_output": logits_out}
        
    
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
        self.pap = PatchAverageProjection(config)
        self.tt_descrite = TransferInterpolationBlock(config)

    def forward(self, patch_tokens: th.Tensor,
                cls_token: th.Tensor,
                timestamps: th.Tensor,
                tmask: Optional[th.BoolTensor | th.LongTensor]=None):

        B = cls_token.size(0)
        z0 = self.encoder(cls_token).squeeze()
        zt_flow = self.flow_model(z0, timestamps, self.cfg.time_chunk_size)
        zt_descrite = self.tt_descrite(timestamps)
        if zt_descrite.ndim == 2:
            zt_descrite = zt_descrite[None, ...].repeat(B, 1, 1)
        pp = self.pap(patch_tokens)
        zt_descrite = (pp.view(B, 1, -1) * zt_descrite)
        zt = self.tt_fuse(zt_flow, zt_descrite)
        xt = self.decoder(zt)
        return self.fuse_head(patch_tokens, xt, tmask, self.cfg.time_chunk_size)
#============================<(Perceptual Fusion Modeling Part)>=============================================================


#============================ViT Modelling Part=============================================================
class PatchEmbedding(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(PatchEmbedding, self).__init__()
        self.cfg = config
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
        (B, C, W, H) = image.shape
        embeddings = self.patch_embed(image)
        tokens = th.cat([embeddings, self.cls.view(1, 1, -1).repeat(B, 1, 1)], dim=1)
        intermediates: Dict[str, th.Tensor | Tuple[th.Tensor]] = {} if get_intermediates else None
        for idx, block in enumerate(self.blocks):
            bout = block(tokens, use_cache=False)
            tokens = bout.last_features
            if intermediates is not None:
                intermediates[idx] = bout.hidden_features

        return {"patch_tokens": tokens[:, :-1, :],
                "cls_token": tokens[:, -1, :],
                "intermediates": intermediates}


class PatchAverageProjection(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(PatchAverageProjection, self).__init__()
        self.projection = nn.Sequential(nn.Linear(config.vfeatures, config.lfeatures),
                                        FiltrationBlock(config.lfeatures))

    def forward(self, patch_tokens: th.Tensor):
        x = patch_tokens.mean(dim=1)
        x = self.projection(x)
        return x
        

#============================ViT Modelling Part=============================================================


@dataclass
class WPTInput:
    logits: Optional[TensorType["B", "Cls"]]=None
    channels: Optional[TensorType["B", "T", "C"]]=None
    embeddings: Optional[TensorType["B", "C"]]=None

@dataclass 
class WPTOutput(ModelOutput):
    loss:                       Optional[th.FloatTensor]=None
    patch_tokens:               Optional[th.Tensor]=None
    cls_token:                  Optional[th.Tensor]=None
    intermediates:              Optional[th.Tensor]=None
    temporal_cls_tokens:        Optional[th.Tensor]=None
    channels_output:            Optional[th.Tensor]=None
    logits_output:              Optional[th.Tensor]=None
    
class WeightedPerceptualTransferModel(PreTrainedModel):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(WeightedPerceptualTransferModel, self).__init__(config)
        self.config = config
        self.visual = VisualTransformer(config)
        self.ptrnasnet = PerceptualTransferModel(config)

    def forward(self, image:            TensorType["B", "W", "H", "C"],
                timestamps:            Optional[TensorType["B", "Time"]]=None,
                inter_timestamps:      Optional[TensorType["Time"]]=None,
                inter_steps:            Optional[int | th.LongTensor | List[int]]=None,
                get_vit_intermediates:  bool=False,
                verbose: bool=True):
        output = self.visual(image, get_vit_intermediates)
        if timestamps is not None:
            cls_token = output["cls_token"][:, None, :]
            temporal_out = self.ptrnasnet(output["patch_tokens"], 
                                cls_token,
                                timestamps=timestamps)
            output.update(temporal_out)
        if inter_timestamps is not None:
            if inter_steps is not None:
                output.update(self._forward_inter_impl(output["patch_tokens"],
                                                    inter_timestamps,
                                                    inter_steps,
                                                    verbose))
            else:
                output.update(self._forward_inter_impl(output["patch_tokens"],
                                                    inter_timestamps, 
                                                    verbose=verbose))
        return WPTOutput(**output)

    def _approximate_impl(self, patch_tokens: TensorType["B", "S", "C"],
                            timestamps: th.FloatTensor):
        z0 = self.ptrnasnet.tt_descrite(timestamps[0])
        bpp = self.ptrnasnet.pap(patch_tokens)
        (B, C) = bpp.shape
        z0 = (bpp.view(B, 1, C) * z0.view(1, 1, C))
        x0 = self.ptrnasnet.decoder(z0)
        return self.ptrnasnet(patch_tokens, x0, timestamps)
        
    def approximate(self, image: TensorType["B", "W", "H", "C"],
                        timestamps: th.FloatTensor):
            output = self.visual(image)
            approx_output = self._approximate_impl(output["patch_tokens"], timestamps)
            output.update(approx_output)
            return WPTOutput(**output)
    
    def interpolate(self, image: TensorType["B", "W", "H", "C"],
                        timestamps: th.FloatTensor,
                        steps: int | th.LongTensor | List[int]=100,
                        verbose: bool=True):
            output = self.visual(image)
            patch_tokens = output["patch_tokens"]
            if isinstance(steps, (th.LongTensor, list)):
                assert (len(steps) == (timestamps.shape[0] + 1))
            (tcls_full_stack, channels_full_stack, logits_full_stack) = [], [], []
            pbar = range(1, len(timestamps))
            if verbose:
                pbar = tqdm(pbar, 
                            desc="Interpolating Values...",
                            colour="green",
                            ascii=":-")
            for tidx in pbar:
                lsteps = (steps if isinstance(steps, int) else steps[tidx - 1])
                times = th.linspace(timestamps[tidx - 1], timestamps[tidx], lsteps)
                fuse_output = self._approximate_impl(patch_tokens, times)
                tcls_full_stack.append(fuse_output["temporal_cls_tokens"])
                channels_full_stack.append(fuse_output["channels_output"])
                logits = fuse_output["logits_output"]
                if logits is not None:
                    logits_full_stack.append(logits)
                    
            output.update({"temporal_cls_tokens": th.cat(tcls_full_stack, dim=1),
                            "channels_output": th.cat(channels_full_stack, dim=1),
                            "logits_output": th.cat(logits_full_stack, dim=1) 
                                            if len(logits_full_stack) != 0 else None})
            return WPTOutput(**output)



# ========================<metrics & losses>==============================================
class WPTCriterionModel(nn.Module):
    def __init__(self, config: WeightedPerceptualTransferConfig):
        super(WPTCriterionModel, self).__init__()
        self.reg_loss = nn.MSELoss()
        if (config.n_classes is not None) and (config.n_classes != 0):
            self.cls_loss = nn.CrossEntropyLoss()
            (self.tp, self.tn) = (0.0, 0.0)
            (self.fp, self.fn) = (0.0, 0.0)
    def _get_classification_stats(self, logits: th.tensor, 
                    labels: th.Tensor, 
                    tau: float=0.45):
        preds = (logits > tau).float()
        self.tp += ((preds == 1) & (labels == 1)).sum().float().item()
        self.tn += ((preds == 0) & (labels == 0)).sum().float().item()
        self.fp += ((preds == 1) & (labels == 0)).sum().float().item()
        self.fn += ((preds == 0) & (labels == 1)).sum().float().item()

    def get_classification_stats(self, empty: bool=True):
        output = {"tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn}
        if empty:
            for k in output.keys():
                setattr(self, k, 0)
        return output
        
    def forward(self, wpt_output: WPTOutput, 
                channels: th.Tensor,
                logits: Optional[th.Tensor]=None):
        output = dict()
        loss = th.tensor(0)
        reg_loss = self.reg_loss(wpt_output["channels_output"], channels)
        output.update({"reg": reg_loss})
        loss += reg_loss
        if logits is not None:
            if wpt_output["logits_output"] is not None:
                logits_loss = self.cls_loss(wpt_output["logits_output"], logits)
                output.update({"classification": logits_loss})
                loss += logits_loss
            else:
                warn("logits loss will work only if " \
                "config.n_classes is not None or not equale to 0")
        output.update({"loss": loss})
        return output

        
if __name__ == "__main__":
    config = WeightedPerceptualTransferConfig(in_channels=3, 
                                            out_channels=13,
                                            visual_features=312,
                                            image_size=448,
                                            time_chunk_size=54,
                                            n_classes=32)
    # tgrid = TransferInterpolationBlock(config)
    # times = th.normal(0, 1, (10, 100))
    # tfeatures = tgrid(times)
    # print(tfeatures.shape)
    model = WeightedPerceptualTransferModel(config)
    data = th.normal(0, 1, (10, 3, 448, 448))
    # times = th.linspace(0, 1, 1000)
    # out = model(data, times)

    times = th.linspace(0.012, 0.3, 100)
    out = model.approximate(data, times)
    print(out.logits_output.shape,
        out.channels_output.shape,
        out.patch_tokens.shape,
        out.temporal_cls_tokens.shape)
    
    # print(fout.fused_features.shape, fout.waves.shape)
    
