# Copyright (c) Facebook, Inc. and its affiliates.
from .instantiate import instantiate
from .lazy import LazyCall, LazyConfig

__all__ = [
    "LazyCall",
    "LazyConfig",
    "instantiate",
]

assert __all__ == sorted(__all__)
