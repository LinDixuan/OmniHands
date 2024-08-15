import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
import einops
from ..dfm_networks.positional_encoding import (
    PositionEmbeddingSine1D,
)
from ..dfm_networks.loss import Joint2DLoss, ManoLoss, DynamicManoLoss
from ..components.pose_transformer import TransformerDecoder

class DynamicFusionModule(nn.Module):
    def __init__(
        self,
        mano_layer,
        pose_feat_size=512,
        shape_feat_size=512,
        mano_neurons=[1024, 512],
    ):
        super(DynamicFusionModule, self).__init__()

        # 6D representation of rotation matrix
        self.pose6d_size = 16 * 6
        self.mano_pose_size = 16 * 3

        # Base Regression Layers
        mano_base_neurons = [pose_feat_size] + mano_neurons
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
            zip(mano_base_neurons[:-1], mano_base_neurons[1:])
        ):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.LeakyReLU(inplace=True))
        self.mano_base_layer = nn.Sequential(*base_layers)
        # Pose layers
        self.pose_reg = nn.Linear(mano_base_neurons[-1], self.pose6d_size)
        # Shape layers
        self.shape_reg = nn.Linear(shape_feat_size, 10)
        # Camera layers
        self.cam_reg = nn.Linear(mano_base_neurons[-1], 3)
        self.rel_reg = nn.Linear(mano_base_neurons[-1] * 2, 3)
        self.mano_layer = mano_layer

    def mano_transform(self, mano_pose, mano_shape, hand_type='right'):
        # mano_pose: axis    [bs x 48]  flat_mean = False
        global_orient = mano_pose[:, :3]
        hand_pose = mano_pose[:, 3:48]
        if self.mano_layer.layer[hand_type].pose_mean.device != hand_pose.device:
            self.mano_layer.layer[hand_type] = self.mano_layer.layer['right'].to(hand_pose.device)
        outputs = self.mano_layer.layer[hand_type](betas=mano_shape,
                                                   hand_pose=hand_pose, global_orient=global_orient)

        verts = outputs.vertices
        joints = torch.FloatTensor(self.mano_layer.sh_joint_regressor).to(verts) @ verts
        return verts, joints

    def forward(
        self,
        pose_feat
    ):
        pose_features = self.mano_base_layer(pose_feat)

        pred_mano_shape = self.shape_reg(pose_feat)
        pred_mano_pose_6d = self.pose_reg(pose_features)
        pred_cam = self.cam_reg(pose_features)
        pose_features = einops.rearrange(pose_features, '(b s) c -> b (s c)', s=2)
        pred_rel = self.rel_reg(pose_features)
        pred_mano_pose_rotmat = (
            rot6d2mat(pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        )
        pred_mano_pose = (
            mat2aa(pred_mano_pose_rotmat.view(-1, 3, 3))
            .contiguous()
            .view(-1, self.mano_pose_size)
        )
        pred_verts, pred_joints = self.mano_transform(
            pred_mano_pose, pred_mano_shape
        )

        pred_mano_results = {
            "verts3d": pred_verts,
            "joints3d": pred_joints,
            "mano_shape": pred_mano_shape,
            "mano_pose": pred_mano_pose,
            "cam": pred_cam,
            "mano_pose6d":pred_mano_pose_6d
        }

        pred_mano_results_split = {}
        for k in pred_mano_results.keys():
            pred = pred_mano_results[k]
            bs = pred.shape[0]
            pred = pred.reshape([bs//2, 2, *pred.shape[1:]])
            pred_right = pred[:, 0]
            pred_left = pred[:, 1]
            pred_mano_results_split[f"{k}_left"] = pred_left
            pred_mano_results_split[f"{k}_right"] = pred_right
        pred_mano_results_split["root_rel"] = pred_rel

        joints_right, joints_left = pred_mano_results_split['joints3d_right'], pred_mano_results_split['joints3d_left']
        verts_right, verts_left = pred_mano_results_split['verts3d_right'], pred_mano_results_split['verts3d_left']
        root_right, root_left = joints_right[:, 9:10, :].clone(), joints_left[:, 9:10, :].clone()
        verts_w_right = verts_right - root_right
        verts_w_left = verts_left - root_left
        joints_w_right = joints_right - root_right
        joints_w_left = joints_left - root_left

        verts_w_left[:, :, 0] = -verts_w_left[:, :, 0]
        joints_w_left[:, :, 0] = -joints_w_left[:, :, 0]
        verts_w_left = verts_w_left + pred_rel.unsqueeze(1)
        joints_w_left = joints_w_left + pred_rel.unsqueeze(1)
        pred_mano_results_split['joints3d_world_right'] = joints_w_right
        pred_mano_results_split['verts3d_world_right'] = verts_w_right
        pred_mano_results_split['joints3d_world_left'] = joints_w_left
        pred_mano_results_split['verts3d_world_left'] = verts_w_left

        return pred_mano_results_split


class TimeHead(nn.Module):
    def __init__(self,
                 cfg,
                 mano_layer,
                 channels=1024,
                 mano_neurons=[1024, 512],
                 ):
        super(TimeHead, self).__init__()

        self.cfg = cfg
        self.seq_len = cfg.MODEL.SEQ_LEN
        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_cam', init_cam)

        self.temporal_position_embedding = PositionEmbeddingSine1D(
            num_pos_feats=channels
        )
        self.temp_decoder = TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=1024,
            dim_head=64,
        )
        self.mano_branch = DynamicFusionModule(
            mano_layer,
            pose_feat_size=channels,
            shape_feat_size=channels,
            mano_neurons=mano_neurons,
        )

    def forward(self, hand_token, batch_seq, train):
        """
        Args:
            hand_token: (bs x T x 2) x C
        """
        batch = {}
        for k in batch_seq.keys():
            batch[k] = batch_seq[k][:, self.seq_len // 2]

        hand_token = einops.rearrange(hand_token, '(b t s) c -> (b s) c t', s=2, t=self.seq_len)
        hand_pos = self.temporal_position_embedding(hand_token)
        hand_token = (hand_token + hand_pos).permute(0, 2, 1).contiguous()
        token_input = torch.zeros_like(hand_token[:, 0:1, 0:1])
        token_out = self.temp_decoder(token_input, context=hand_token).squeeze(1)  # (B S) C
        pred_mano_result = self.mano_branch(
            token_out
        )
        pred_mano_result = self.solve_cam_t(pred_mano_result, batch, train)
        return pred_mano_result

    def solve_cam_t(self, pred_mano_params, batch, train):
        pred_cam_right = pred_mano_params[f'cam_right']
        pred_cam_left = pred_mano_params[f'cam_left']

        device = pred_mano_params[f'joints3d_right'].device
        dtype = pred_mano_params[f'joints3d_right'].dtype

        batch_size = pred_cam_right.shape[0]
        mean_cam = self.init_cam.expand(batch_size, -1)
        pred_cam_right = pred_cam_right + mean_cam
        pred_cam_left = pred_cam_left + mean_cam
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(pred_cam_right.shape[0], 2, device=device, dtype=dtype)
        cam_t_right = torch.stack([pred_cam_right[:, 1],
                                  pred_cam_right[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam_right[:, 0] + 1e-9)],
                                 dim=-1)
        pred_mano_params[f'pred_cam_t_right'] = cam_t_right  # bs x 3

        cam_t_right = cam_t_right.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        j2d_right = perspective_projection(pred_mano_params[f'joints3d_right'],
                                                   translation=cam_t_right,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        pred_mano_params['joints2d_right'] = j2d_right.clone()


        cam_t_left = torch.stack([pred_cam_left[:, 1],
                                  pred_cam_left[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam_left[:, 0] + 1e-9)],
                                  dim=-1)
        pred_mano_params[f'pred_cam_t_right'] = cam_t_left  # bs x 3

        cam_t_left = cam_t_left.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        j2d_left = perspective_projection(pred_mano_params[f'joints3d_left'],
                                                translation=cam_t_left,
                                                focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        pred_mano_params['joints2d_left'] = j2d_left.clone()

        j2d_right_glb, j2d_left_glb = self.calc_glb_2d(batch, pred_mano_params[f'joints3d_right'].clone(),
                                                       pred_mano_params[f'joints3d_left'].clone(),
                                                       cam_t_right.clone(), cam_t_left.clone(),
                                                       pred_mano_params['root_rel'].clone())
        pred_mano_params['joints2d_glb_right'] = j2d_right_glb
        pred_mano_params['joints2d_glb_left'] = j2d_left_glb

        if not train:
            j2d_left[..., 0] = -j2d_left[..., 0]

            bbox_right = batch["bbox_right"]
            bbox_left = batch["bbox_left"]
            bbox_full = batch["bbox_full"]
            IoU = calculate_iou(bbox_right, bbox_left)

            j2d_right = (j2d_right + 0.5) * bbox_right[:, None, 2:4]
            j2d_left = (j2d_left + 0.5) * bbox_left[:, None, 2:4]
            j2d_right = j2d_right + bbox_right[:, None, 0:2] - bbox_full[:, None, 0:2]
            j2d_left = j2d_left + bbox_left[:, None, 0:2] - bbox_full[:, None, 0:2]

            pred_mano_params['joints2d_world_right'] = j2d_right.clone()
            pred_mano_params['joints2d_world_left'] = j2d_left.clone()

            j3d_right, j3d_left = pred_mano_params["joints3d_world_right"].clone(), \
                                  pred_mano_params["joints3d_world_left"].clone()

            root_dist = torch.linalg.norm(j3d_left[:, 9, :] - j3d_right[:, 9, :], dim=-1)
            j2d_align = torch.cat([j2d_right, j2d_left], dim=1)
            j3d_align = torch.cat([j3d_right, j3d_left], dim=1)

            full_size = bbox_full[0, 2:].detach().cpu().numpy()

            focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * full_size.max()

            cam_k = np.array([[focal_length, 0., full_size[0] / 2],
                             [0., focal_length, full_size[1] / 2],
                             [0., 0., 1.]], dtype=np.float32)

            j2d_right = j2d_right.float().detach().cpu().numpy()
            j2d_left = j2d_left.float().detach().cpu().numpy()
            j3d_right = j3d_right.float().detach().cpu().numpy()
            j3d_left = j3d_left.float().detach().cpu().numpy()
            j2d_align = j2d_align.float().detach().cpu().numpy()
            j3d_align = j3d_align.float().detach().cpu().numpy()

            cam_left = []
            cam_right = []
            cam_align = []
            for idx in range(j3d_right.shape[0]):
                ret_left, r_left, t_left, _ = cv2.solvePnPRansac(j3d_left[idx], j2d_left[idx], cam_k, np.zeros(4))
                ret_right, r_right, t_right, _ = cv2.solvePnPRansac(j3d_right[idx], j2d_right[idx], cam_k, np.zeros(4))
                ret_align, r_align, t_align, _ = cv2.solvePnPRansac(j3d_align[idx], j2d_align[idx], cam_k, np.zeros(4))
                assert ret_left and ret_right and ret_align
                cam_left.append(t_left.squeeze(1))
                cam_right.append(t_right.squeeze(1))
                cam_align.append(t_align.squeeze(1))

            cam_left = np.stack(cam_left, axis=0)
            cam_right = np.stack(cam_right, axis=0)
            cam_align = np.stack(cam_align, axis=0)

            cam_t_right = torch.from_numpy(cam_right).to(device)
            cam_t_left = torch.from_numpy(cam_left).to(device)
            cam_aligned_right = torch.from_numpy(cam_align).to(device)
            cam_aligned_left = torch.from_numpy(cam_align).to(device)
            align_mask = (IoU < 0.01) + (root_dist > 0.2)
            cam_aligned_right[align_mask] = cam_t_right[align_mask]
            cam_aligned_left[align_mask] = cam_t_left[align_mask]
            pred_mano_params["cam_aligned_right"] = cam_aligned_right
            pred_mano_params["cam_aligned_left"] = cam_aligned_left

        return pred_mano_params

    def calc_glb_2d(self, batch, j3d_right, j3d_left, cam_right, cam_left, root_rel):
        j3d_left[..., 0] = -j3d_left[..., 0]
        cam_left[..., 0] = -cam_left[..., 0]
        root_right = j3d_right[:, 9, :].clone()
        root_left = j3d_left[:, 9, :].clone()
        trans_r2l = root_left - root_right
        trans_l2r = root_right - root_left
        cam_left_align = trans_l2r + cam_right + root_rel
        cam_right_align = trans_r2l + cam_left - root_rel

        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(j3d_right.shape[0], 2,
                                                                device=j3d_right.device, dtype=j3d_right.dtype)

        j2d_l = perspective_projection(j3d_left,
                                       translation=cam_left_align,
                                       focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        j2d_r = perspective_projection(j3d_right,
                                       translation=cam_right_align,
                                       focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        bbox_r, bbox_l, bbox_f = batch['bbox_right'].clone(), batch['bbox_left'].clone(), batch['bbox_full'].clone()

        j2d_r = (j2d_r + 0.5) * bbox_l[:, 2:4].unsqueeze(1) + bbox_l[:, :2].unsqueeze(1)
        j2d_l = (j2d_l + 0.5) * bbox_r[:, 2:4].unsqueeze(1) + bbox_r[:, :2].unsqueeze(1)
        j2d_right_glb = (j2d_r - bbox_f[:, :2].unsqueeze(1)) / bbox_f[:, 2:4].unsqueeze(1) - 0.5
        j2d_left_glb = (j2d_l - bbox_f[:, :2].unsqueeze(1)) / bbox_f[:, 2:4].unsqueeze(1) - 0.5
        return j2d_right_glb, j2d_left_glb


def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix."""
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def quat2aa(quaternion):
    """Convert quaternion vector to angle axis of rotation."""
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def mat2quat(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector"""
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def rot6d2mat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def mat2aa(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector"""

    def convert_points_to_homogeneous(points):
        if not torch.is_tensor(points):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(points))
            )
        if len(points.shape) < 2:
            raise ValueError(
                "Input must be at least a 2D tensor. Got {}".format(points.shape)
            )

        return F.pad(points, (0, 1), "constant", 1.0)

    if rotation_matrix.shape[1:] == (3, 3):
        rotation_matrix = convert_points_to_homogeneous(rotation_matrix)
    quaternion = mat2quat(rotation_matrix)
    aa = quat2aa(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


class HandHeatmapLayer(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21):
        """
        Args:
            inr_res: input image size
            joint_nb: hand joint num
        """
        super(HandHeatmapLayer, self).__init__()

        # hand head
        self.out_res = roi_res
        self.joint_nb = joint_nb

        self.betas = nn.Parameter(torch.ones((self.joint_nb, 1), dtype=torch.float32))

        center_offset = 0.5
        vv, uu = torch.meshgrid(
            torch.arange(self.out_res).float(), torch.arange(self.out_res).float()
        )
        uu, vv = uu + center_offset, vv + center_offset
        self.register_buffer("uu", uu / self.out_res)
        self.register_buffer("vv", vv / self.out_res)

        self.softmax = nn.Softmax(dim=2)

    def spatial_softmax(self, latents):
        latents = latents.view((-1, self.joint_nb, self.out_res**2))
        latents = latents * self.betas
        heatmaps = self.softmax(latents)
        heatmaps = heatmaps.view(-1, self.joint_nb, self.out_res, self.out_res)
        return heatmaps

    def generate_output(self, heatmaps):
        predictions = torch.stack(
            (
                torch.sum(torch.sum(heatmaps * self.uu, dim=2), dim=2),
                torch.sum(torch.sum(heatmaps * self.vv, dim=2), dim=2),
            ),
            dim=2,
        )
        return predictions

    def forward(self, latent):
        heatmap = self.spatial_softmax(latent)
        prediction = self.generate_output(heatmap)
        return prediction

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

def perspective_projection(points,
                           translation,
                           focal_length,
                           camera_center=None,
                           rotation=None):
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]