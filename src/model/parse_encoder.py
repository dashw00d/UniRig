from dataclasses import dataclass

from .michelangelo.get_model import get_encoder as get_encoder_michelangelo
from .michelangelo.get_model import AlignedShapeLatentPerceiver
from .michelangelo.get_model import get_encoder_simplified as get_encoder_michelangelo_encoder
from .michelangelo.get_model import ShapeAsLatentPerceiverEncoder

@dataclass(frozen=True)
class _MAP_MESH_ENCODER:
    ptv3obj = "ptv3obj"
    michelangelo = AlignedShapeLatentPerceiver
    michelangelo_encoder = ShapeAsLatentPerceiverEncoder

MAP_MESH_ENCODER = _MAP_MESH_ENCODER()


def get_mesh_encoder(**kwargs):
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'ptv3obj':
        from .pointcept.models.PTv3Object import get_encoder as get_encoder_ptv3obj
        return get_encoder_ptv3obj(**kwargs)
    if __target__ == 'michelangelo':
        return get_encoder_michelangelo(**kwargs)
    if __target__ == 'michelangelo_encoder':
        return get_encoder_michelangelo_encoder(**kwargs)
    raise AssertionError(
        f"expect: [ptv3obj,michelangelo,michelangelo_encoder], found: {__target__}"
    )
