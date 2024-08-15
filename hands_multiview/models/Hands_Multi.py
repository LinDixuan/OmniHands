import os.path
import posixpath
import random

import numpy as np
import torch
import pytorch_lightning as pl
import pyrender
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils.geometry import aa_to_rotmat, perspective_projection, distance_map, merge_bbox, bbox_2_square, combine_box, \
    proj_orth, add_map
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_mano_head
from . import MANO
from .heads.inter_head import interhand_head
from ..utils.renderer import Renderer, cam_crop_to_full
from .components.t_cond_mlp import ResidualMLP, create_simple_mlp
from .components.pose_transformer import *
from .modules.graph_utils import get_meshsample_layer
from .heads.aggregate_head import AggregateHead
from ..datasets.relighted_mano import MANO as rel_mano
from ..datasets.relighted_mano import axis_to_Rmat
from .dfm_losses import ManoLoss

import cv2 as cv

log = get_pylogger(__name__)
import einops

LIGHT_BLUE = (0.1255, 0.8157, 0.9176)
LIGHT_RED = (0.98039216, 0.47058824, 0.49411765)

class Hands_Multi(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.log_dir = 'logs_mul'
        self.exp_name = cfg.exp_name
        self.cfg = cfg
        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}

        self.mano = MANO(**mano_cfg)
        self.rel_mano = rel_mano(use_pca=False)
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        # Create MANO head
        self.mano_head = build_mano_head(cfg)
        self.map_head = create_simple_mlp(input_dim=11, hidden_dims=[128],
                                          output_dim=1280, activation=torch.nn.GELU())
        self.aggregate_head = AggregateHead(cfg, self.rel_mano)
        for name, param in self.aggregate_head.named_parameters():
            if 'cam' in name:
                param.requires_grad = False
        self.seq_len = cfg.MODEL.SEQ_LEN

        # Define loss functions
        self.mano_loss = ManoLoss(
            lambda_verts3d=1e3,
            lambda_joints3d=1e3,
            lambda_joints2d=1e3,
            lambda_manopose=10,
            lambda_manoshape=1,
            lambda_consistence=1e3,
            lambda_rel_joints=1e3,
            lambda_close=1e3,
            lambda_root=1e3,
            seq_len=self.seq_len,
            loss_base="maxmse"
        )

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        #self.freeze_pretrained_params()

        self.persp_renderer = Renderer(self.cfg, faces=self.mano.faces)
        self.val_outputs = []
        self.val_batches = []
        self.gpu = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_cam', init_cam)

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None and param.requires_grad == True:
                print('on after backward')
                print(param.shape)
                print(name)

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        all_params += list(self.time_head.parameter())
        all_params += list(self.map_head.parameter())
        return all_params

    def freeze_pretrained_params(self):
        """for param in self.mano_head.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.mano_head.eval()
        self.backbone.eval()"""

    def configure_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                      # lr=self.cfg.TRAIN.LR,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50_000, gamma=0.7)
        return [optimizer], [scheduler]

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        img_right = batch['img_right']  # bs x T x 256 x 256 x 3  normalized
        img_left = batch['img_left']

        batch_size, T = img_right.shape[0:2]
        assert T == self.seq_len
        img = torch.stack([img_right, img_left], dim=1)     # bs x 2 x T x 256 x 256 x 3
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio

        x = einops.rearrange(img, 'b s t w h c -> (b s t) c w h')

        conditioning_feats = self.backbone(x[:, :, :, 32:-32])  # (B x S x T) x 1280 x 16 x 12

        posmap_R, posmap_L = add_map(batch['bbox_right'].flatten(0, 1),
                                     batch['bbox_left'].flatten(0, 1), activate=True)   # (B x T) x 16 x 12 x 21

        posmap_R = self.map_head(posmap_R).flatten(1, 2)    # (B x T) x (16x12) x 1280
        posmap_L = self.map_head(posmap_L).flatten(1, 2)

        cond_feat = einops.rearrange(conditioning_feats, '(b s t) c h w -> (b t) s (h w) c', s=2, t=T)
        cond_feat_right = cond_feat[:, 0]
        cond_feat_left = cond_feat[:, 1]
        cond_feat_right = cond_feat_right + posmap_R     # (B x T) x (16 x 12) x 1280
        cond_feat_left = cond_feat_left + posmap_L

        hand_tokens = self.mano_head(cond_feat_right, cond_feat_left)  # (B x T x S) x 1024

        pred_mano_params = self.aggregate_head(hand_tokens, batch, train)

        for hand_side in ['right', 'left']:
            if f"mano_param_{hand_side}" in batch.keys():
                gt_mano_param = batch[f"mano_param_{hand_side}"].flatten(0, 1)
                gt_verts3d, gt_joints3d = self.aggregate_head.mano_branch.mano_transform(gt_mano_param[:, :48],
                                                                                         gt_mano_param[:, 48:])
                batch[f"verts3d_{hand_side}"] = gt_verts3d.view(batch_size, T,
                                                                *gt_verts3d.shape[1:])  # bs x T x 778 x 3
                batch[f"joints3d_{hand_side}"] = gt_joints3d.view(batch_size, T,
                                                                  *gt_joints3d.shape[1:])  # bs x T x 21 x 3

        return pred_mano_params


    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        batch_size, T = batch["img_right"].shape[0:2]
        losses = {}
        gts = {}
        for k in batch.keys():
            gts[k] = batch[k].flatten(0, 1)
        mano_total_loss, mano_losses = self.mano_loss.compute_loss(output, gts)
        for k in mano_losses.keys():
            if train:
                losses[k] = mano_losses[k]
            else:
                losses[k] = mano_losses[k].detach().cpu()

        loss_all = mano_total_loss
        if not train:
            loss_all = loss_all.detach().cpu()
        losses["loss"] = loss_all
        output["losses"] = losses
        return loss_all


    def forward(self, batch: Dict) -> list:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """

        optimizer = self.optimizers(use_pl_optimizer=True)
        scheduler = self.lr_schedulers()

        output = self.forward_step(batch, train=True)

        loss = self.compute_loss(batch, output, train=True)

        # Error if Nan
        if torch.isnan(loss):
            print(output['losses'])
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL,
                                                error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        scheduler.step()
        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    @pl.utilities.rank_zero.rank_zero_only
    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:

        with torch.no_grad():
            output = self.forward_step(batch, train=False)
            loss = self.compute_loss(batch, output, train=False)

        mpjpe, mpvpe, mrrpe = self.eval_step(batch, output)

        output[f'mpjpe'], output[f'mpvpe'], output[f'mrrpe']= \
            torch.FloatTensor([mpjpe]), torch.FloatTensor([mpvpe]), torch.FloatTensor([mrrpe])
        for k in output.keys():
            if isinstance(output[k], torch.Tensor):
                output[k] = output[k].cpu()

        output['bbox_full'] = batch['bbox_full'].flatten(0, 1).cpu()
        self.val_outputs.append(output)
        if self.cfg.TRAIN.MULTITHREAD:
            if batch['img_full'].dim() == 5:
                center_img = batch['img_full'].flatten(0, 1)
            else:
                center_img = batch['img_full']
            self.val_batches.append(center_img)
        return output

    @pl.utilities.rank_zero.rank_zero_only
    def on_validation_epoch_end(self):

        summary_writer = self.logger.experiment
        outputs = self.val_outputs
        current_epoch = self.current_epoch
        current_step = self.global_step

        avg_loss = torch.stack([x['losses']['loss'] for x in outputs]).mean()
        avg_joint = torch.stack([x[f'mpjpe'] for x in outputs]).mean()
        avg_vert = torch.stack([x[f'mpvpe'] for x in outputs]).mean()
        avg_root = torch.stack([x[f'mrrpe'] for x in outputs]).mean()

        summary_writer.add_scalar('val' + '/' + f'mpjpe', avg_joint.item(), current_step)
        summary_writer.add_scalar('val' + '/' + f'mpvpe', avg_vert.item(), current_step)
        summary_writer.add_scalar('val' + '/' + f'mrrpe', avg_root.item(), current_step)
        for loss_name in outputs[-1]['losses'].keys():
            loss = torch.stack([x['losses'][loss_name] for x in outputs]).mean()
            summary_writer.add_scalar('val' + '/' + f'{loss_name}', loss.item(), current_step)


        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val_acc', avg_joint, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if not os.path.exists(f'/workspace/hamer_intertime/{self.log_dir}/logs/{self.exp_name}'):
            os.makedirs(f'/workspace/hamer_intertime/{self.log_dir}/logs/{self.exp_name}', exist_ok=True)
        with open(f'/workspace/hamer_intertime/{self.log_dir}/logs/{self.exp_name}/acc_log.txt', 'a') as f:
            f.writelines(f'epoch:{current_epoch} step:{current_step // 1000}K\n')
            f.writelines(f'losses: {float(avg_loss.item()):.3f}\n')
            f.writelines(f"joint:{float(avg_joint.item()):.3f}  "
                         f"vert:{float(avg_vert.item()):.3f} "
                         f"root:{float(avg_root.item()):.3f}  \n")

        if self.cfg.TRAIN.MULTITHREAD:
            test_img = []
            verts_right = []
            cam_right = []
            verts_left = []
            cam_left = []
            full_sizes = []

            for tb in range(4):
                test_id = random.randint(0, len(self.val_batches)-1)
                test_img.append(self.val_batches[test_id])
                test_out = self.val_outputs[test_id]
                verts_right.append(test_out['verts3d_world_right'])
                cam_right.append(test_out['cam_aligned_right'])
                verts_left.append(test_out['verts3d_world_left'])
                cam_left.append(test_out['cam_aligned_left'])
                full_sizes.append(test_out['bbox_full'][:, 2:4])

            test_img = torch.cat(test_img, dim=0)
            verts_right = torch.cat(verts_right, dim=0).detach().cpu().numpy()
            cam_right = torch.cat(cam_right, dim=0).detach().cpu().numpy()
            verts_left = torch.cat(verts_left, dim=0).detach().cpu().numpy()
            cam_left = torch.cat(cam_left, dim=0).detach().cpu().numpy()
            full_sizes = torch.cat(full_sizes, dim=0).detach().cpu().numpy()

            save_path = f'/workspace/hamer_intertime/{self.log_dir}/render/{self.exp_name}/e{current_epoch:03d}_s{self.global_step:07d}'
            os.makedirs(save_path, exist_ok=True)
            for i in range(verts_right.shape[0]):
                all_verts = [verts_right[i], verts_left[i]]
                all_cam_t = [cam_right[i], cam_left[i]]
                all_right = [np.array([1], dtype=np.float32), np.array([0], dtype=np.float32)]
                scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * full_sizes[i].max()
                py_renderer = pyrender.OffscreenRenderer(viewport_width=int(full_sizes[i][0]),
                                                         viewport_height=int(full_sizes[i][1]),
                                                         point_size=1.0)
                misc_args = dict(
                    mesh_base_color=[LIGHT_BLUE, LIGHT_RED],
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                    renderer=py_renderer
                )
                cam_view = self.persp_renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t,
                                                                    render_res=(int(full_sizes[i][0]), int(full_sizes[i][1])),
                                                                    is_right=all_right, **misc_args)
                #cam_view = cv.resize(cam_view, (256, 256))

                # Overlay image
                img_cv2 = test_img[i].detach().cpu().numpy().copy()
                img_cv2 = cv.resize(img_cv2, (int(full_sizes[i][0]), int(full_sizes[i][1])))
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

                img_output = np.concatenate([img_cv2, 255 * input_img_overlay[:, :, ::-1]], axis=1)
                cv.imwrite(os.path.join(save_path, f'rend_{i:03d}.jpg'), img_output)
                py_renderer.delete()
        
        self.val_batches = []
        self.val_outputs = []


    @torch.no_grad()
    def eval_step(self, batch_data, output=None):
        MPJPE = 0
        MPVPE = 0
        num = 0
        if output is None:
            output = self.forward_step(batch_data)

        for hand_side in ['right', 'left']:
            joints_pred = output[f'joints3d_{hand_side}'].clone()
            verts_pred = output[f'verts3d_{hand_side}'].clone()

            joints_gt = batch_data[f'joints3d_{hand_side}'].clone().flatten(0, 1)
            verts_gt = batch_data[f'verts3d_{hand_side}'].clone().flatten(0, 1)

            mpjpe, mpvpe, w = eval_error(joints_gt, verts_gt,
                                         joints_pred, verts_pred)
            MPJPE = MPJPE + mpjpe.item() * w
            MPVPE = MPVPE + mpvpe.item() * w
            num = num + int(joints_gt.shape[0])

        MPJPE = MPJPE / num
        MPVPE = MPVPE / num

        root_pred = output["joints3d_world_right"][:, 9, :] - output["joints3d_world_left"][:, 9, :]
        root_gt = batch_data["joints3d_world_right"][:, :, 9, :] - \
                  batch_data["joints3d_world_left"][:, :, 9, :]

        MRRPE = torch.linalg.norm(root_gt.flatten(0, 1) - root_pred, dim=-1).mean() * 1000

        return MPJPE, MPVPE, MRRPE

    def get_merge_cam(self, local_boxes, full_boxes, cam, flip=False):
        local_size = local_boxes[:, 2:].clone()
        full_size = full_boxes[:, 2:].clone()

        local_center = torch.stack([local_boxes[:, 0] - full_boxes[:, 0] + local_boxes[:, 2] / 2,
                                    local_boxes[:, 1] - full_boxes[:, 1] + local_boxes[:, 3] / 2], dim=-1)
        if flip:
            local_center[:, 0] = full_boxes[:, 2] - local_center[:, 0]

        focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * full_size[:, 0]

        pred_cam_aligned = cam_crop_to_full(cam, local_center, local_size[:, 0], full_size, focal_length)
        return pred_cam_aligned


    @pl.utilities.rank_zero.rank_zero_only
    def visualize_gt(self, batch, output):
        save_mesh = True
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        MEAN = 255. * np.array(IMAGE_MEAN)
        STD = 255. * np.array(IMAGE_STD)
        for k in batch.keys():
            batch[k] = batch[k].flatten(0, 1)
        verts_w_right = output['verts3d_world_right']
        verts_w_left = output['verts3d_world_left']
        joints_w_right = output['joints3d_world_right']
        joints_w_left = output['joints3d_world_left']
        imgs_full = batch['img_full']
        imgs_right = batch['img_right']
        imgs_left = batch['img_left']
        j2d_gt_right = batch['joints2d_right']
        j2d_gt_left = batch['joints2d_left']
        verts3d_right = batch['verts3d_right']
        verts3d_left = batch['verts3d_left']
        j2d_pd_right = output['joints2d_right']
        j2d_pd_left = output['joints2d_left']
        bboxes_right = batch['bbox_right'].int().cpu().numpy()
        bboxes_left = batch['bbox_left'].int().cpu().numpy()

        pose_right_pred = output["mano_pose_right"]
        pose_left_pred = output["mano_pose_left"]
        right_aa = pose_right_pred[:, :3]
        left_aa = pose_left_pred[:, :3]
        left_aa[:, 1:3] = -left_aa[:, 1:3]
        right_rmat = axis_to_Rmat(right_aa)
        left_rmat = axis_to_Rmat(left_aa)
        rmat_inv_l = torch.linalg.inv(left_rmat.transpose(-2, -1))
        rmat_inv_r = torch.linalg.inv(right_rmat.transpose(-2, -1))
        v3d_left_regul = verts_w_left - joints_w_left[:, 9:10]
        v3d_right_regul = verts_w_right - joints_w_left[:, 9:10]
        v3d_left_regul = v3d_left_regul @ rmat_inv_l
        v3d_right_regul = v3d_right_regul @ rmat_inv_l

        imgs_full = imgs_full.detach().cpu().numpy().copy()
        imgs_right = imgs_right.detach().cpu().numpy().copy()
        imgs_left = imgs_left.detach().cpu().numpy().copy()
        for n_c in range(3):
            imgs_right[:, :, :, n_c] = imgs_right[:, :, :, n_c] * STD[n_c] + MEAN[n_c]
            imgs_left[:, :, :, n_c] = imgs_left[:, :, :, n_c] * STD[n_c] + MEAN[n_c]

        if save_mesh:
            save_path_ply = '/workspace/hamer_intertime/gt_test/mesh'
            os.makedirs(save_path_ply, exist_ok=True)

        save_path_img = '/workspace/hamer_intertime/gt_test/img'

        os.makedirs(save_path_img, exist_ok=True)
        for i in range(imgs_full.shape[0]):
            img = imgs_full[i].copy()
            """bbox_right = bboxes_right[i]
            bbox_left = bboxes_left[i]
            
            cv.rectangle(img, (bbox_right[0], bbox_right[1]),
                         (bbox_right[0]+bbox_right[2], bbox_right[1]+bbox_right[3]),
                         (0, 0, 255), 2)
            cv.rectangle(img, (bbox_left[0], bbox_left[1]),
                         (bbox_left[0]+bbox_left[2], bbox_left[1]+bbox_left[3]),
                         (255, 0, 0), 2)"""
            cv.imwrite(os.path.join(save_path_img, f'B_{i:03d}_img_full.jpg'), img)

        for i in range(imgs_right.shape[0]):
            img = imgs_right[i].copy()
            for j in range(j2d_gt_right[i].shape[0]):
                cv.circle(img, (
                    int(j2d_gt_right[i][j, 0] * 256 + 128),
                    int(j2d_gt_right[i][j, 1] * 256 + 128)), 2,
                          (255, 0, 0), 2)
                """cv.circle(img, (
                    int(j2d_pd_right[i][j, 0] * 256 + 128),
                    int(j2d_pd_right[i][j, 1] * 256 + 128)), 2,
                          (0, 0, 255), 2)"""

            if save_mesh:
                #save_mesh_to_ply(verts_right[i], os.path.join(save_path_ply, f"{i:03d}_right.ply"), "right")
                save_mesh_to_ply(verts3d_right[i], os.path.join(save_path_ply, f"{i:03d}_right_world.ply"), "right")
            cv.imwrite(os.path.join(save_path_img, f'B_{i:03d}_img_right.jpg'), img)

        for i in range(imgs_left.shape[0]):
            img = imgs_left[i].copy()
            for j in range(j2d_gt_left[i].shape[0]):
                cv.circle(img, (
                    int(j2d_gt_left[i][j, 0] * 256 + 128),
                    int(j2d_gt_left[i][j, 1] * 256 + 128)), 2,
                          (255, 0, 0), 2)
                """cv.circle(img, (
                    int(j2d_pd_left[i][j, 0] * 256 + 128),
                    int(j2d_pd_left[i][j, 1] * 256 + 128)), 2,
                          (0, 0, 255), 2)"""

            if save_mesh:
                #save_mesh_to_ply(verts_left[i], os.path.join(save_path_ply, f"{i:03d}_left.ply"), "right")
                save_mesh_to_ply(verts3d_left[i], os.path.join(save_path_ply, f"{i:03d}_left_world.ply"), "left")
            cv.imwrite(os.path.join(save_path_img, f'B_{i:03d}_img_left.jpg'), img)

    def inference_token_forward(self, batch):
        img_right = batch['img_right']  # bs x T x 256 x 256 x 3  normalized
        img_left = batch['img_left']

        img = torch.stack([img_right, img_left], dim=1)  # bs x 2 x T x 256 x 256 x 3
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio

        x = einops.rearrange(img, 'b s w h c -> (b s) c w h')

        conditioning_feats = self.backbone(x[:, :, :, 32:-32])  # (B x S) x 1280 x 16 x 12

        posmap_R, posmap_L = add_map(batch['bbox_right'],
                                     batch['bbox_left'], activate=True)  # B x 16 x 12 x 21

        posmap_R = self.map_head(posmap_R).flatten(1, 2)  # (B x T) x (16x12) x 1280
        posmap_L = self.map_head(posmap_L).flatten(1, 2)

        cond_feat = einops.rearrange(conditioning_feats, '(b s) c h w -> b s (h w) c', s=2)
        cond_feat_right = cond_feat[:, 0]
        cond_feat_left = cond_feat[:, 1]
        cond_feat_right = cond_feat_right + posmap_R  # B x (16 x 12) x 1280
        cond_feat_left = cond_feat_left + posmap_L

        hand_tokens = self.mano_head(cond_feat_right, cond_feat_left)  # (B x S) x 1024

        return hand_tokens

    def inference_temp_forward(self, hand_tokens, batch):
        pred_mano_params = self.time_head(hand_tokens, batch, False)
        return pred_mano_params

@torch.no_grad()
def eval_error(joints_gt, verts_gt,
               joints_pred, verts_pred,
               root_idx=9):
    root_gt = joints_gt[:, root_idx:root_idx + 1]
    joints_gt = joints_gt - root_gt
    verts_gt = verts_gt - root_gt

    root_pred = joints_pred[:, root_idx:root_idx + 1]
    joints_pred = joints_pred - root_pred
    verts_pred = verts_pred - root_pred

    length_gt = torch.linalg.norm(joints_gt[:, 9:10] - joints_gt[:, 0:1], dim=-1).unsqueeze(-1).clamp_min(
        0.01)  # bs x 1 x 1
    length_pred = torch.linalg.norm(joints_pred[:, 9:10] - joints_pred[:, 0:1], dim=-1).unsqueeze(-1).clamp_min(
        0.01)  # bs x 1 x 1
    joints_pred = joints_pred * length_gt / length_pred  # bs x J x 3
    verts_pred = verts_pred * length_gt / length_pred  # bs x J x 3

    MPJPE = torch.linalg.norm(joints_pred - joints_gt, dim=-1).mean() * 1000
    MPVPE = torch.linalg.norm(verts_pred - verts_gt, dim=-1).mean() * 1000

    weight = root_gt.shape[0]

    return MPJPE, MPVPE, weight


def check_invalid_values(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def calculate_iou(box1, box2):
    """
    box : bs x 4
    """

    # 计算交集的左上角和右下角坐标
    inter_left = torch.max(box1[:, 0], box2[:, 0])
    inter_top = torch.max(box1[:, 1], box2[:, 1])
    inter_right = torch.min(box1[:, 0] + box1[:, 2], box2[:, 0] + box2[:, 2])
    inter_bottom = torch.min(box1[:, 1] + box1[:, 3], box2[:, 1] + box2[:, 3])

    # 计算交集的宽度和高度
    inter_width = torch.clamp(inter_right - inter_left, min=0)
    inter_height = torch.clamp(inter_bottom - inter_top, min=0)

    # 计算交集的面积
    inter_area = inter_width * inter_height

    # 计算两个框的面积
    area_box1 = box1[:, 2] * box1[:, 3]
    area_box2 = box2[:, 2] * box2[:, 3]

    # 计算 IoU
    iou = inter_area / (area_box1 + area_box2 - inter_area)

    return iou