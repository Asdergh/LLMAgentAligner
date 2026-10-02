import torch as th
import torch.nn as nn
from typing import Optional
from ...layers import FiltrationBlock, get_activation



__BACKBONES__ = dict()

def register_backbone(name: str, overwrite: bool=False):
    """Clss decorator that registers a backbone class under ``name``.
    Args:
        name: Unique key of the backbone inside the registry.
        overwrite: Replace an already registered class instead of raising an error.
    Notes:
        The class is stored, not instantiated. It must be an ``nn.Module`` that
        builds the model inside ``__init__``, exposes an integer ``embedding_dim``
        and returns a tensor of shape ``[B, S, embedding_dim]`` from ``forward``.
    Raises:
        TypeError: If the decorated object is not an ``nn.Module`` subclass.
        ValueError: If ``name`` is already registered and ``overwrite`` is False.
    """
    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            raise TypeError(f"Only nn.Module subclasses can be registered as backbones, got {cls}.")
        if name in __BACKBONES__ and not overwrite:
            raise ValueError(f"Backbone '{name}' is already registered. "
                            "Pass overwrite=True to replace it.")
        __BACKBONES__[name] = cls
        return cls
    return decorator

def get_backbone(
    name: str, 
    features_dim: int, 
    backbone_kwargs: Optional[dict]=None,
    **kwargs
):
    """Instantiate a registered backbone and wrap it into ``BackBoneWrapper``.
    Args:
        name: Key of the backbone registered with ``register_backbone``.
        features_dim: Output size of the projection head.
        backbone_kwargs: Keyword arguments for the backbone class constructor.
        **kwargs: Extra arguments for ``BackBoneWrapper`` (``activation``,
            ``pre_dp``, ``post_dp``, ``filtration``).
    Notes:
        Constructor arguments of the backbone and of the wrapper are passed
        separately, so they never collide.
    Returns:
        BackBoneWrapper: Wrapper around the built backbone.
    Raises:
        KeyError: If ``name`` is not registered.
        ValueError: If the backbone class failed to instantiate.
    """
    cls = __BACKBONES__.get(name, None)
    if cls is None:
        raise KeyError(f"Backbone '{name}' is not registered. "
                        f"Available backbones: {list(__BACKBONES__.keys())}")
    try:
        backbone = cls(**(backbone_kwargs or {}))
    except Exception as e:
        raise ValueError(f"Failed to build backbone '{name}': {e}") from e
    return BackBoneWrapper(features_dim=features_dim,
                            backbone=backbone,
                            **kwargs)

class BackBoneWrapper(nn.Module):
    """Frozen backbone with a trainable projection head.
    Data flow::
        x -> backbone -> head -> filter -> out
        [B, S, C] -> [B, S, features_dim]
    Args:
        features_dim: Output size of the head.
        backbone: Ready ``nn.Module`` with an integer ``embedding_dim`` attribute.
        activation: Name of the activation for get_activation``.
        pre_dp: Dropout probability before the linear layer.
        post_dp: Dropout probability after the linear layer.
        filtration: If True, ``FiltrationBlock(features_dim)`` is applied to
            the output of the head.
    Notes:
        ``head = Dropout(pre_dp) -> Linear(embedding_dim, features_dim) ->
        Dropout(post_dp) -> activation`` is applied token-wise to the last
        dimension, the sequence axis is not pooled.
        Backbone parameters are frozen and the backbone stays in eval mode
        even after ``.train()``.
        ``FiltrationBlock`` receives a tensor of shape ``[B, S, features_dim]``.
    Raises:
        ValueError: If ``backbone`` is not an ``nn.Module`` or has no integer
            ``embedding_dim``.
    """
    def __init__(self, 
                features_dim: int,
                backbone: nn.Module,
                activation: str="relu",
                pre_dp: float=0.01,
                post_dp: float=0.00,
                filtration: bool=False):
        super(BackBoneWrapper, self).__init__()
        if not isinstance(backbone, nn.Module):
            raise ValueError(f"Unsupported backbone type: {type(backbone)}. "
                            "Expected nn.Module.")
        embedding_dim = getattr(backbone, "embedding_dim", None)
        if not isinstance(embedding_dim, int):
            raise ValueError(f"Backbone {type(backbone).__name__} must expose "
                            "an integer attribute 'embedding_dim'.")
        self._embedding_dim = embedding_dim
        self.head = nn.Sequential(nn.Dropout(pre_dp),
                                    nn.Linear(embedding_dim, features_dim),
                                    nn.Dropout(post_dp),
                                    get_activation(activation))
        self._filter = FiltrationBlock(features_dim) if filtration else None
        backbone.requires_grad_(False)
        self._backbone = backbone.eval()

    def forward(self, x: th.Tensor, **kwargs):
        """Project backbone tokens to ``features_dim``.
        Args:
            x: Input of the backbone, passed as is.
            **kwargs: Extra keyword arguments for the backbone,
                e.g. ``attention_mask``.
        Notes:
            The backbone is called once, without any pre/post processing.
            Its output shape is validated before the head is applied.
        Returns:
            th.Tensor: Features of shape ``[B, S, features_dim]``.
        Raises:
            ValueError: If the backbone output is not a tensor of shape
                ``[B, S, embedding_dim]``.
        """
        x = self._backbone(x, **kwargs)
        if not isinstance(x, th.Tensor):
            raise ValueError(f"Backbone must return th.Tensor of shape [B, S, C], got {type(x)}.")
        if x.dim() != 3 or x.size(-1) != self._embedding_dim:
            raise ValueError(f"Backbone must return shape [B, S, {self._embedding_dim}], "
                            f"got {list(x.shape)}.")
        x = self.head(x)
        if self._filter is not None:
            x = self._filter(x)
        return x
