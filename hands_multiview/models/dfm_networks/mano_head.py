import torch
from torch import nn
from torch.nn import functional as F


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


class DynamicFusionModule(nn.Module):
    def __init__(
        self,
        mano_layer_right,
        mano_layer_left,
        pose_feat_size=512,
        shape_feat_size=512,
        mano_neurons=[1024, 512],
        coord_change_mat=None,
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
        # forward dynamic Pose layers
        self.fw_dynamic_pose_reg = nn.Linear(mano_base_neurons[-1], self.pose6d_size)
        # backward dynamic Pose layers
        self.bw_dynamic_pose_reg = nn.Linear(mano_base_neurons[-1], self.pose6d_size)
        # confidences
        self.confidence_reg = nn.Linear(mano_base_neurons[-1], 1)

        self.mano_layer_right = mano_layer_right
        self.mano_layer_left = mano_layer_left

        if coord_change_mat is not None:
            self.register_buffer("coord_change_mat", coord_change_mat)
        else:
            self.coord_change_mat = None

    def mano_transform(self, mano_side, mano_pose, mano_shape):
        if (mano_side == 0).all():
            verts, joints = self.mano_layer_right(
                th_pose_coeffs=mano_pose, th_betas=mano_shape
            )
        elif (mano_side == 1).all():
            verts, joints = self.mano_layer_left(
                th_pose_coeffs=mano_pose, th_betas=mano_shape
            )
        else:
            verts, joints = [], []
            for i, side in enumerate(mano_side):
                if side == 0:
                    vert, joint = self.mano_layer_right(
                        th_pose_coeffs=mano_pose[i : i + 1],
                        th_betas=mano_shape[i : i + 1],
                    )
                elif side == 1:
                    vert, joint = self.mano_layer_left(
                        th_pose_coeffs=mano_pose[i : i + 1],
                        th_betas=mano_shape[i : i + 1],
                    )
                else:
                    raise Exception("Unknown MANO side!")
                verts.append(vert)
                joints.append(joint)
            verts = torch.cat(verts, dim=0)
            joints = torch.cat(joints, dim=0)
        return verts, joints

    def get_center_prediction(
        self,
        pred_mano_shape,
        pred_mano_pose_6d,
        fw_dynamic_pred_mano_pose_6d,
        bw_dynamic_pred_mano_pose_6d,
        mano_side,
        batch,
        T,
        confidences=None,
        coverage=None,
    ):
        cidx = T // 2
        # computer culmulative motion
        fw_dynamic_pred_mano_pose_6d = fw_dynamic_pred_mano_pose_6d.view(
            batch, T, self.pose6d_size
        )[
            :, :cidx
        ]  # Bx(T/2)x48
        cum_fw_dynamic_pred_mano_pose_6d = (
            fw_dynamic_pred_mano_pose_6d.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        )  # Bx(T/2)x48
        bw_dynamic_pred_mano_pose_6d = bw_dynamic_pred_mano_pose_6d.view(
            batch, T, self.pose6d_size
        )[
            :, cidx + 1 :
        ]  # Bx(T/2)x48
        cum_bw_dynamic_pred_mano_pose_6d = bw_dynamic_pred_mano_pose_6d.cumsum(
            dim=1
        )  # Bx(T/2)x48
        # warp pose
        expanded_pred_mano_pose_6d = pred_mano_pose_6d.view(
            batch, T, self.pose6d_size
        )  # BxTx48
        fw_pred_mano_pose_6d = (
            expanded_pred_mano_pose_6d[:, :cidx] + cum_fw_dynamic_pred_mano_pose_6d
        )  # Bx(T/2)x48
        bw_pred_mano_pose_6d = (
            expanded_pred_mano_pose_6d[:, cidx + 1 :] + cum_bw_dynamic_pred_mano_pose_6d
        )  # Bx(T/2)x48
        # run mano
        t_mano_sides = mano_side.view(batch, T).mean(dim=1)  # B
        t_pred_mano_shape = pred_mano_shape  # Bx10
        # compute the averged mano parameters
        t_pred_mano_pose_6d = torch.cat(
            [
                fw_pred_mano_pose_6d,
                expanded_pred_mano_pose_6d[:, cidx : cidx + 1],
                bw_pred_mano_pose_6d,
            ],
            dim=1,
        )
        if confidences is None:
            t_pred_mano_pose_6d = t_pred_mano_pose_6d.mean(dim=1)  # Bx48
        else:
            # learned confidence
            confidences = confidences.view(batch, T, 1)
            confidences = F.softmax(confidences, dim=1)
            t_pred_mano_pose_6d = (t_pred_mano_pose_6d * confidences).sum(dim=1)

        t_pred_mano_pose_rotmat = (
            rot6d2mat(t_pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        )
        t_pred_mano_pose = (
            mat2aa(t_pred_mano_pose_rotmat.view(-1, 3, 3))
            .contiguous()
            .view(-1, self.mano_pose_size)
        )
        t_pred_verts, t_pred_joints = self.mano_transform(
            t_mano_sides, t_pred_mano_pose, t_pred_mano_shape
        )
        return (
            t_pred_mano_shape,
            t_pred_mano_pose,
            t_pred_mano_pose_rotmat,
            t_pred_verts,
            t_pred_joints,
        )

    def forward(
        self,
        pose_feat,
        shape_feat,
        mano_side=None,
        mano_params=None,
        roots3d=None,
        batch=None,
        T=None,
        coverage=None,
    ):
        pose_features = self.mano_base_layer(pose_feat)
        pred_mano_shape = self.shape_reg(shape_feat)
        expanded_pred_mano_shape = (
            pred_mano_shape.unsqueeze(1).repeat(1, T, 1).flatten(0, 1)
        )
        pred_mano_pose_6d = self.pose_reg(pose_features)
        pred_mano_pose_rotmat = (
            rot6d2mat(pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        )
        pred_mano_pose = (
            mat2aa(pred_mano_pose_rotmat.view(-1, 3, 3))
            .contiguous()
            .view(-1, self.mano_pose_size)
        )
        pred_verts, pred_joints = self.mano_transform(
            mano_side, pred_mano_pose, expanded_pred_mano_shape
        )

        if mano_params is not None:
            gt_mano_shape = mano_params[:, self.mano_pose_size :]
            gt_mano_pose = mano_params[:, : self.mano_pose_size].contiguous()
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(
                -1, 16, 3, 3
            )
            gt_verts, gt_joints = self.mano_transform(
                mano_side, gt_mano_pose, gt_mano_shape
            )

            gt_mano_results = {
                "verts3d": gt_verts,
                "joints3d": gt_joints,
                "mano_shape": gt_mano_shape,
                "mano_pose": gt_mano_pose_rotmat,
            }
        else:
            gt_mano_results = None

        if roots3d is not None:  # evaluation
            pred_mano_results = {}
            cidx = T // 2
            roots3d = roots3d.view(batch, T, 3)[:, cidx : cidx + 1]
            fw_dynamic_pred_mano_pose_6d = self.fw_dynamic_pose_reg(
                pose_features
            )  # forward motion
            bw_dynamic_pred_mano_pose_6d = self.bw_dynamic_pose_reg(
                pose_features
            )  # backward motion
            confidences = self.confidence_reg(pose_features)  # confidences
            (
                t_pred_mano_shape,
                t_pred_mano_pose,
                t_pred_mano_pose_rotmat,
                pred_verts,
                pred_joints,
            ) = self.get_center_prediction(
                pred_mano_shape,
                pred_mano_pose_6d,
                fw_dynamic_pred_mano_pose_6d,
                bw_dynamic_pred_mano_pose_6d,
                mano_side,
                batch,
                T,
                confidences=confidences,
                coverage=coverage,
            )
            confidences = confidences.view(batch, T)
            confidences = F.softmax(confidences, dim=1)
            pred_mano_results.update(
                {
                    "mano_shape": t_pred_mano_shape,
                    "mano_pose": t_pred_mano_pose_rotmat,
                    "mano_pose_aa": t_pred_mano_pose,
                    "confidences": confidences,
                }
            )

            pred_verts3d, pred_joints3d = pred_verts + roots3d, pred_joints + roots3d
            if self.coord_change_mat is not None:
                pred_verts3d = pred_verts3d.matmul(self.coord_change_mat)
                pred_joints3d = pred_joints3d.matmul(self.coord_change_mat)
            pred_mano_results.update(
                {
                    "verts3d": pred_verts3d,
                    "joints3d": pred_joints3d,
                }
            )
        else:
            pred_mano_results = {
                "verts3d": pred_verts,
                "joints3d": pred_joints,
                "mano_shape": expanded_pred_mano_shape,
                "mano_pose": pred_mano_pose_rotmat,
                "expanded_mano_pose_aa": pred_mano_pose.view(
                    batch, T, self.mano_pose_size
                ),
            }

            fw_dynamic_pred_mano_pose_6d = self.fw_dynamic_pose_reg(
                pose_features
            )  # forward motion
            bw_dynamic_pred_mano_pose_6d = self.bw_dynamic_pose_reg(
                pose_features
            )  # backward motion
            confidences = self.confidence_reg(pose_features)  # confidences
            # forward 1 step
            fw_pred_mano_pose_6d = pred_mano_pose_6d + fw_dynamic_pred_mano_pose_6d
            fw_pred_mano_pose_rotmat = (
                rot6d2mat(fw_pred_mano_pose_6d.view(-1, 6))
                .view(-1, 16, 3, 3)
                .contiguous()
            )
            fw_pred_mano_pose = (
                mat2aa(fw_pred_mano_pose_rotmat.view(-1, 3, 3))
                .contiguous()
                .view(-1, self.mano_pose_size)
            )
            fw_pred_verts, fw_pred_joints = self.mano_transform(
                mano_side, fw_pred_mano_pose, expanded_pred_mano_shape
            )
            # backward 1 step
            bw_pred_mano_pose_6d = pred_mano_pose_6d + bw_dynamic_pred_mano_pose_6d
            bw_pred_mano_pose_rotmat = (
                rot6d2mat(bw_pred_mano_pose_6d.view(-1, 6))
                .view(-1, 16, 3, 3)
                .contiguous()
            )
            bw_pred_mano_pose = (
                mat2aa(bw_pred_mano_pose_rotmat.view(-1, 3, 3))
                .contiguous()
                .view(-1, self.mano_pose_size)
            )
            bw_pred_verts, bw_pred_joints = self.mano_transform(
                mano_side, bw_pred_mano_pose, expanded_pred_mano_shape
            )
            # end to end ouput
            (
                t_pred_mano_shape,
                t_pred_mano_pose,
                t_pred_mano_pose_rotmat,
                t_pred_verts,
                t_pred_joints,
            ) = self.get_center_prediction(
                pred_mano_shape,
                pred_mano_pose_6d,
                fw_dynamic_pred_mano_pose_6d,
                bw_dynamic_pred_mano_pose_6d,
                mano_side,
                batch,
                T,
                confidences=confidences,
            )
            pred_mano_results.update(
                {
                    "fw_verts3d": fw_pred_verts,
                    "fw_joints3d": fw_pred_joints,
                    "fw_mano_pose": fw_pred_mano_pose_rotmat,
                    "expanded_fw_mano_pose_aa": fw_pred_mano_pose.view(
                        batch, T, self.mano_pose_size
                    ),
                    "bw_verts3d": bw_pred_verts,
                    "bw_joints3d": bw_pred_joints,
                    "bw_mano_pose": bw_pred_mano_pose_rotmat,
                    "expanded_bw_mano_pose_aa": bw_pred_mano_pose.view(
                        batch, T, self.mano_pose_size
                    ),
                    "t_verts3d": t_pred_verts,
                    "t_joints3d": t_pred_joints,
                    "t_mano_pose": t_pred_mano_pose_rotmat,
                }
            )

        return pred_mano_results, gt_mano_results
