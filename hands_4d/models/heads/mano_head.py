import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder, TransformerCrossAttn_CASA, Transformer
from ..components.t_cond_mlp import create_simple_mlp
from ..modules.graph_utils import get_meshsample_layer

def build_mano_head(cfg):
    mano_head_type = cfg.MODEL.MANO_HEAD.get('TYPE', 'hamer')
    if  mano_head_type == 'transformer_decoder':
        return MANOTransformerDecoderHead(cfg)
    else:
        raise ValueError('Unknown MANO head type: {}'.format(mano_head_type))

class MANOTransformerDecoderHead(nn.Module):
    """ Cross-attention based MANO Transformer decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.MANO_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'

        """self.inter_attn = TransformerCrossAttn_CASA(dim=1280,
                                                    depth=3,
                                                    heads=16,
                                                    dim_head=80,
                                                    mlp_dim=1280)"""

        self.inter_attn = Transformer(dim=1280,
                                      depth=4,
                                      heads=16,
                                      dim_head=80,
                                      mlp_dim=1280)
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = (transformer_args | dict(cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER))
        self.transformer = TransformerDecoder(
            **transformer_args
        )

    def forward(self, cond_right, cond_left, **kwargs):

        batch_size = cond_right.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        cond_feat = torch.cat([cond_right, cond_left], dim=1)
        cond_feat = self.inter_attn(cond_feat)
        token_num = cond_feat.shape[1] // 2
        cond_right = cond_feat[:, :token_num]
        cond_left = cond_feat[:, token_num:]
        #cond_right = self.inter_attn(cond_right, context=cond_left)
        #cond_left = self.inter_attn(cond_left, context=cond_right)

        x = torch.stack([cond_right, cond_left], dim=1)
        x = einops.rearrange(x, 'b s l c -> (b s) l c')
        # Input token to transformer is zero token
        token = torch.zeros(batch_size * 2, 1, 1).to(x.device)

        # Pass through transformer
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1)  # (B, C)

        return token_out
