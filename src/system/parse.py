import torch
from torch.optim import Optimizer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter

def get_system(**kwargs) -> LightningModule:
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'ar':
        from .ar import ARSystem
        return ARSystem(**kwargs)
    if __target__ == 'skin':
        from .skin import SkinSystem
        return SkinSystem(**kwargs)
    raise AssertionError(f"expect: [ar,skin], found: {__target__}")

def get_writer(**kwargs) -> BasePredictionWriter:
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'ar':
        from .ar import ARWriter
        return ARWriter(**kwargs)
    if __target__ == 'skin':
        from .skin import SkinWriter
        return SkinWriter(**kwargs)
    raise AssertionError(f"expect: [ar,skin], found: {__target__}")
