from ssd.modeling import registry
from .vgg import VGG
from .vggm import VGGM

__all__ = ['build_backbone', 'VGG']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
