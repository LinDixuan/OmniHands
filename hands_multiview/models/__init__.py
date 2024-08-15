from .mano_wrapper import MANO
from .Hands_Multi import Hands_Multi

import torch

def load_from_ckpt(checkpoint_path, model_cfg):
    from ..configs import get_config
    model_cfg = get_config(model_cfg, update_cachedir=True)
    model = Hands_Multi(model_cfg)
    model_state_dict = torch.load(checkpoint_path)
    model.load_state_dict(model_state_dict, strict=False)
    return model, model_cfg
