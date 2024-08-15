import torch
import torch.nn.functional as F


def maxMSE(input, target):
    # input: ...xNx3 or ...xN
    if input.shape[-1] == 3:
        error2 = torch.sum((input - target) ** 2, dim=-1)
    else:
        raise NotImplementedError()
    error4 = error2**2
    return torch.mean(torch.sum(error4, dim=-1) / torch.sum(error2, dim=-1))


class Joint2DLoss:
    def __init__(self, lambda_joints2d):
        super(Joint2DLoss, self).__init__()
        self.lambda_joints2d = lambda_joints2d

    def compute_loss(self, preds, gts):
        final_loss = 0.0
        joint_losses = {}
        if type(preds) != list:
            preds = [preds]
        num_stack = len(preds)
        mask = gts > 0.0
        # in case no valid point
        if not mask.any():
            joint_losses["hm_joints2d_loss"] = 0.0
            return final_loss, joint_losses

        for i, pred in enumerate(preds):
            joints2d_loss = self.lambda_joints2d * F.mse_loss(pred[mask], gts[mask])
            final_loss += joints2d_loss
            if i == num_stack - 1:
                joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
        final_loss /= num_stack
        return final_loss, joint_losses


class ManoLoss:
    def __init__(
        self,
        lambda_verts3d=None,
        lambda_joints3d=None,
        lambda_manopose=None,
        lambda_manoshape=None,
        lambda_regulshape=None,
        lambda_regulpose=None,
        loss_base=None,
    ):
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_manopose = lambda_manopose
        self.lambda_manoshape = lambda_manoshape
        self.lambda_regulshape = lambda_regulshape
        self.lambda_regulpose = lambda_regulpose
        self.loss_base = loss_base

    def compute_loss(self, preds, gts):
        loss_fn = F.mse_loss
        if self.loss_base == "maxmse":
            loss_fn = maxMSE

        final_loss = 0
        mano_losses = {}
        if type(preds) != list:
            preds = [preds]
        num_preds = len(preds)
        for i, pred in enumerate(preds):
            if self.lambda_verts3d is not None and "verts3d" in gts:
                mesh3d_loss = self.lambda_verts3d * loss_fn(
                    pred["verts3d"], gts["verts3d"]
                )
                final_loss += mesh3d_loss
                if i == num_preds - 1:
                    mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
            if self.lambda_joints3d is not None and "joints3d" in gts:
                joints3d_loss = self.lambda_joints3d * loss_fn(
                    pred["joints3d"], gts["joints3d"]
                )
                final_loss += joints3d_loss
                if i == num_preds - 1:
                    mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
            if self.lambda_manopose is not None and "mano_pose" in gts:
                pose_param_loss = self.lambda_manopose * F.mse_loss(
                    pred["mano_pose"], gts["mano_pose"]
                )
                final_loss += pose_param_loss
                if i == num_preds - 1:
                    mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
            if self.lambda_manoshape is not None and "mano_shape" in gts:
                shape_param_loss = self.lambda_manoshape * F.mse_loss(
                    pred["mano_shape"], gts["mano_shape"]
                )
                final_loss += shape_param_loss
                if i == num_preds - 1:
                    mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
            if self.lambda_regulshape:
                shape_regul_loss = self.lambda_regulshape * F.mse_loss(
                    pred["mano_shape"], torch.zeros_like(pred["mano_shape"])
                )
                final_loss += shape_regul_loss
                if i == num_preds - 1:
                    mano_losses[
                        "regul_manoshape_loss"
                    ] = shape_regul_loss.detach().cpu()
            if self.lambda_regulpose:
                pose_regul_loss = self.lambda_regulpose * F.mse_loss(
                    pred["mano_pose"][:, 3:], torch.zeros_like(pred["mano_pose"][:, 3:])
                )
                final_loss += pose_regul_loss
                if i == num_preds - 1:
                    mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
        final_loss /= num_preds
        mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        return final_loss, mano_losses


class DynamicManoLoss:
    def __init__(
        self,
        lambda_dynamic_verts3d=None,
        lambda_dynamic_joints3d=None,
        lambda_dynamic_manopose=None,
        lambda_dynamic_manoshape=None,
        lambda_end2end_verts3d=None,
        lambda_end2end_joints3d=None,
        lambda_end2end_manopose=None,
        lambda_end2end_manoshape=None,
        lambda_temporal_verts3d=None,
        lambda_temporal_joints3d=None,
        lambda_temporal_manopose=None,
        lambda_temporal_manoshape=None,
        temporal_constrained=False,
        loss_base=None,
    ):
        self.lambda_dynamic_verts3d = lambda_dynamic_verts3d
        self.lambda_dynamic_joints3d = lambda_dynamic_joints3d
        self.lambda_dynamic_manopose = lambda_dynamic_manopose
        self.lambda_end2end_verts3d = lambda_end2end_verts3d
        self.lambda_end2end_joints3d = lambda_end2end_joints3d
        self.lambda_end2end_manopose = lambda_end2end_manopose
        if temporal_constrained:
            self.lambda_temporal_verts3d = lambda_temporal_verts3d
            self.lambda_temporal_joints3d = lambda_temporal_joints3d
            self.lambda_temporal_manopose = lambda_temporal_manopose
        else:
            self.lambda_temporal_verts3d = None
            self.lambda_temporal_joints3d = None
            self.lambda_temporal_manopose = None
        self.loss_base = loss_base

    def compute_loss(self, preds, gts, batch, T):
        loss_fn = F.mse_loss
        if self.loss_base == "maxmse":
            loss_fn = maxMSE

        final_loss = 0
        dynamic_mano_losses = {}
        if type(preds) != list:
            preds = [preds]
        num_preds = len(preds)
        expanded_gts_verts3d = gts["verts3d"].view(batch, T, *gts["verts3d"].shape[1:])
        expanded_gts_joints3d = gts["joints3d"].view(
            batch, T, *gts["joints3d"].shape[1:]
        )
        expanded_gts_mano_pose = gts["mano_pose"].view(
            batch, T, *gts["mano_pose"].shape[1:]
        )
        for i, pred in enumerate(preds):
            # dynamic mesh
            if (
                (self.lambda_dynamic_verts3d is not None)
                and ("fw_verts3d" in pred)
                and ("verts3d" in gts)
            ):
                expanded_fw_verts3d = pred["fw_verts3d"].view(
                    batch, T, *pred["fw_verts3d"].shape[1:]
                )
                expanded_bw_verts3d = pred["bw_verts3d"].view(
                    batch, T, *pred["bw_verts3d"].shape[1:]
                )
                dynamic_mesh3d_loss = self.lambda_dynamic_verts3d * (
                    loss_fn(expanded_fw_verts3d[:, :-1], expanded_gts_verts3d[:, 1:])
                    + loss_fn(expanded_bw_verts3d[:, 1:], expanded_gts_verts3d[:, :-1])
                )
                final_loss += dynamic_mesh3d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_mano_mesh3d_loss"
                    ] = dynamic_mesh3d_loss.detach().cpu()
            # dynamic joints3d
            if (
                (self.lambda_dynamic_joints3d is not None)
                and ("fw_joints3d" in pred)
                and ("joints3d" in gts)
            ):
                expanded_fw_joints3d = pred["fw_joints3d"].view(
                    batch, T, *pred["fw_joints3d"].shape[1:]
                )
                expanded_bw_joints3d = pred["bw_joints3d"].view(
                    batch, T, *pred["bw_joints3d"].shape[1:]
                )
                dynamic_joints3d_loss = self.lambda_dynamic_joints3d * (
                    loss_fn(expanded_fw_joints3d[:, :-1], expanded_gts_joints3d[:, 1:])
                    + loss_fn(
                        expanded_bw_joints3d[:, 1:], expanded_gts_joints3d[:, :-1]
                    )
                )
                final_loss += dynamic_joints3d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_mano_joints3d_loss"
                    ] = dynamic_joints3d_loss.detach().cpu()
            # dynamic pose
            if (
                (self.lambda_dynamic_manopose is not None)
                and ("fw_mano_pose" in pred)
                and ("mano_pose" in gts)
            ):
                expanded_fw_mano_pose = pred["fw_mano_pose"].view(
                    batch, T, *pred["fw_mano_pose"].shape[1:]
                )
                expanded_bw_mano_pose = pred["bw_mano_pose"].view(
                    batch, T, *pred["bw_mano_pose"].shape[1:]
                )
                dynamic_mano_pose_loss = self.lambda_dynamic_manopose * (
                    F.mse_loss(
                        expanded_fw_mano_pose[:, :-1], expanded_gts_mano_pose[:, 1:]
                    )
                    + F.mse_loss(
                        expanded_bw_mano_pose[:, 1:], expanded_gts_mano_pose[:, :-1]
                    )
                )
                final_loss += dynamic_mano_pose_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_mano_pose_loss"
                    ] = dynamic_mano_pose_loss.detach().cpu()
            # slow mesh
            if (
                (self.lambda_temporal_verts3d is not None)
                and ("fw_verts3d" in pred)
                and ("verts3d" in gts)
            ):
                temporal_mesh3d_loss = self.lambda_temporal_verts3d * (
                    loss_fn(pred["fw_verts3d"], pred["verts3d"])
                    + loss_fn(pred["bw_verts3d"], pred["verts3d"])
                )
                final_loss += temporal_mesh3d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "temporal_mano_mesh3d_loss"
                    ] = temporal_mesh3d_loss.detach().cpu()
            # slow joints3d
            if (
                (self.lambda_temporal_joints3d is not None)
                and ("fw_joints3d" in pred)
                and ("joints3d" in gts)
            ):
                temporal_joints3d_loss = self.lambda_temporal_joints3d * (
                    loss_fn(pred["fw_joints3d"], pred["joints3d"])
                    + loss_fn(pred["bw_joints3d"], pred["joints3d"])
                )
                final_loss += temporal_joints3d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "temporal_mano_joints3d_loss"
                    ] = temporal_joints3d_loss.detach().cpu()
            # slow pose
            if (
                (self.lambda_temporal_manopose is not None)
                and ("fw_mano_pose" in pred)
                and ("mano_pose" in gts)
            ):
                temporal_manopose_loss = self.lambda_temporal_manopose * (
                    F.mse_loss(pred["fw_mano_pose"], pred["mano_pose"])
                    + F.mse_loss(pred["bw_mano_pose"], pred["mano_pose"])
                )
                final_loss += temporal_manopose_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "temporal_manopose_loss"
                    ] = temporal_manopose_loss.detach().cpu()
            # end2end mesh
            if (
                (self.lambda_end2end_verts3d is not None)
                and ("t_verts3d" in pred)
                and ("verts3d" in gts)
            ):
                end2end_mesh3d_loss = self.lambda_end2end_verts3d * loss_fn(
                    pred["t_verts3d"], expanded_gts_verts3d[:, T // 2]
                )
                final_loss += end2end_mesh3d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_end2end_mesh3d_loss"
                    ] = end2end_mesh3d_loss.detach().cpu()
            # end2end joints3d
            if (
                (self.lambda_end2end_joints3d is not None)
                and ("t_joints3d" in pred)
                and ("joints3d" in gts)
            ):
                end2end_joints3d_loss = self.lambda_end2end_joints3d * loss_fn(
                    pred["t_joints3d"], expanded_gts_joints3d[:, T // 2]
                )
                final_loss += end2end_joints3d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_end2end_joints3d_loss"
                    ] = end2end_joints3d_loss.detach().cpu()
            # end2end pose
            if (
                (self.lambda_end2end_manopose is not None)
                and ("t_mano_pose" in pred)
                and ("mano_pose" in gts)
            ):
                end2end_manopose_loss = self.lambda_end2end_manopose * F.mse_loss(
                    pred["t_mano_pose"], expanded_gts_mano_pose[:, T // 2]
                )
                final_loss += end2end_manopose_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_end2end_manopose_loss"
                    ] = end2end_manopose_loss.detach().cpu()
        final_loss /= num_preds
        dynamic_mano_losses["dynamic_mano_total_loss"] = final_loss.detach().cpu()
        return final_loss, dynamic_mano_losses


def batch_encoder_disc_l2_loss(disc_value):
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = (
        torch.sum(fake_disc_value**2) / kb,
        torch.sum((real_disc_value - 1) ** 2) / ka,
    )
    return la, lb, la + lb
