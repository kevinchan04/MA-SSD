from ssd.modeling import registry
from .neckthreemed import NECKTHREEMED


__all__ = ['build_neck', 'NECKTHREEMED'
]


def build_neck(cfg):
    if cfg.MODEL.NECK.NAME != 'NONE':
        return registry.NECKS[cfg.MODEL.NECK.NAME](cfg)
    else:
        return None
