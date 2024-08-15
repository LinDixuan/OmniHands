from .mano_wrapper import MANO
from .Hands4D import Hands4D
import torch

def load_from_ckpt(checkpoint_path, model_cfg):
    from ..configs import get_config
    model_cfg = get_config(model_cfg, update_cachedir=True)
    model = Hands4D(model_cfg)
    model_state_dict = torch.load(checkpoint_path)
    model.load_state_dict(model_state_dict, strict=False)
    return model, model_cfg

def load_model(checkpoint_path, model_cfg):
    from pathlib import Path
    from ..configs import get_config

    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    model = Hands4D.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)

    return model, model_cfg
