import torch
import torch.nn.functional as F

joint_weight = torch.FloatTensor([1, 1, 1, 1, 3,
                                  1, 1, 1, 3,
                                  1, 1, 1, 3,
                                  1, 1, 1, 3,
                                  1, 1, 1, 3])

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
                joint_losses["hm_joints2d_loss"] = joints2d_loss.detach()
        final_loss /= num_stack
        return final_loss, joint_losses


class ManoLoss:
    def __init__(
        self,
        lambda_verts3d=None,
        lambda_joints3d=None,
        lambda_verts2d=None,
        lambda_joints2d=None,
        lambda_manopose=None,
        lambda_manoshape=None,
        lambda_rel_joints=None,
        lambda_close=None,
        lambda_root=None,
        lambda_regulshape=None,
        lambda_regulpose=None,
        loss_base=None,
    ):
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_verts2d = lambda_verts2d
        self.lambda_joints2d = lambda_joints2d
        self.lambda_manopose = lambda_manopose
        self.lambda_manoshape = lambda_manoshape
        self.lambda_rel_joints = lambda_rel_joints
        self.lambda_close = lambda_close
        self.lambda_root = lambda_root
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
            for hand_side in ['right', 'left']:
                if self.lambda_verts3d is not None and f"verts3d_{hand_side}" in gts:
                    pred_root = pred[f"joints3d_{hand_side}"][:, 9:10, :]
                    gt_root = gts[f"joints3d_{hand_side}"][:, 9:10, :]
                    mesh3d_loss = self.lambda_verts3d * loss_fn(
                        pred[f"verts3d_{hand_side}"] - pred_root,
                        gts[f"verts3d_{hand_side}"] - gt_root
                    )
                    final_loss += mesh3d_loss
                    if i == num_preds - 1:
                        mano_losses[f"mano_mesh3d_{hand_side}_loss"] = mesh3d_loss.detach()
                if self.lambda_joints3d is not None and f"joints3d_{hand_side}" in gts:
                    pred_root = pred[f"joints3d_{hand_side}"][:, 9:10, :]
                    gt_root = gts[f"joints3d_{hand_side}"][:, 9:10, :]
                    joints3d_loss = self.lambda_joints3d * loss_fn(
                        pred[f"joints3d_{hand_side}"] - pred_root,
                        gts[f"joints3d_{hand_side}"] - gt_root
                    )
                    final_loss += joints3d_loss
                    if i == num_preds - 1:
                        mano_losses[f"mano_joints3d_{hand_side}_loss"] = joints3d_loss.detach()
                if self.lambda_joints2d is not None and f"joints2d_{hand_side}" in gts:
                    mask = (gts[f"joints2d_{hand_side}"] > -0.5) * (gts[f"joints2d_{hand_side}"] < 0.5)
                    mask = mask.all(dim=-1)
                    if not mask.any():
                        joints2d_loss = torch.zeros_like(joints3d_loss)
                    else:
                        joints2d_loss = self.lambda_joints2d * F.mse_loss(
                            pred[f"joints2d_{hand_side}"][mask], gts[f"joints2d_{hand_side}"][mask]
                        )

                    final_loss += joints2d_loss
                    if i == num_preds - 1:
                        mano_losses[f"mano_joints2d_{hand_side}_loss"] = joints2d_loss.detach()

                if self.lambda_joints2d is not None and f"joints2d_glb_{hand_side}" in gts and gts["inter"].any():
                    inter_mask = gts["inter"] > 0.5
                    mask = inter_mask
                    if not mask.any():
                        joints2d_w_loss = torch.zeros_like(joints3d_loss)
                    else:
                        joints2d_w_loss = self.lambda_joints2d * F.mse_loss(
                            pred[f"joints2d_glb_{hand_side}"][mask], gts[f"joints2d_glb_{hand_side}"][mask]
                        )

                    final_loss += joints2d_w_loss
                    if i == num_preds - 1:
                        mano_losses[f"joints2d_glb_{hand_side}_loss"] = joints2d_w_loss.detach()
                if self.lambda_manopose is not None and f"mano_pose_{hand_side}" in gts:
                    pose_param_loss = self.lambda_manopose * F.mse_loss(
                        pred[f"mano_pose_{hand_side}"], gts[f"mano_pose_{hand_side}"]
                    )
                    final_loss += pose_param_loss
                    if i == num_preds - 1:
                        mano_losses[f"manopose_{hand_side}_loss"] = pose_param_loss.detach()
                if self.lambda_manoshape is not None and f"mano_shape_{hand_side}" in gts:
                    shape_param_loss = self.lambda_manoshape * F.mse_loss(
                        pred[f"mano_shape_{hand_side}"], gts[f"mano_shape_{hand_side}"]
                    )
                    final_loss += shape_param_loss
                    if i == num_preds - 1:
                        mano_losses[f"manoshape_{hand_side}_loss"] = shape_param_loss.detach()
            if self.lambda_rel_joints is not None and gts["inter"].any():
                assert gts["inter"].dim() == 1
                assert pred["verts3d_world_right"].dim() == 3
                inter_mask = gts["inter"] > 0.5
                j3d_right_pred, j3d_left_pred = pred["joints3d_world_right"], pred["joints3d_world_left"]
                j3d_right_gt, j3d_left_gt = gts["joints3d_world_right"], gts["joints3d_world_left"]
                jr_pred = (j3d_right_pred.unsqueeze(2) - j3d_left_pred.unsqueeze(1)).flatten(1, 2)
                jr_gt = (j3d_right_gt.unsqueeze(2) - j3d_left_gt.unsqueeze(1)).flatten(1, 2)
                rel_joints_loss = self.lambda_rel_joints * loss_fn(
                    jr_gt[inter_mask], jr_pred[inter_mask]
                )
                final_loss += rel_joints_loss
                if i == num_preds - 1:
                    mano_losses[
                        "rel_joints_loss"
                    ] = rel_joints_loss.detach()
            if self.lambda_close is not None and gts["inter"].any():
                assert gts["inter"].dim() == 1
                assert pred["verts3d_world_right"].dim() == 3
                inter_mask = gts["inter"] > 0.5
                v3d_right_pred, v3d_left_pred = pred["verts3d_world_right"][inter_mask], pred["verts3d_world_left"][inter_mask]
                v3d_right_gt, v3d_left_gt = gts["verts3d_world_right"][inter_mask], gts["verts3d_world_left"][inter_mask]
                vr_pred = v3d_right_pred.unsqueeze(2) - v3d_left_pred.unsqueeze(1)
                vr_gt = v3d_right_gt.unsqueeze(2) - v3d_left_gt.unsqueeze(1)
                close = (torch.linalg.norm(vr_gt, dim=-1) < 0.005) + (torch.linalg.norm(vr_pred, dim=-1) < 0.005)
                if close.sum() > 1:
                    close_loss = self.lambda_close * F.mse_loss(
                        vr_gt[close], vr_pred[close]
                    )
                else:
                    close_loss = torch.zeros_like(pose_param_loss)
                final_loss += close_loss
                if i == num_preds - 1:
                    mano_losses[
                        "close_loss"
                    ] = close_loss.detach()
            if self.lambda_root is not None:
                inter_mask = gts["inter"] > 0.5
                j3d_right_pred, j3d_left_pred = pred["joints3d_world_right"], pred["joints3d_world_left"]
                j3d_right_gt, j3d_left_gt = gts["joints3d_world_right"], gts["joints3d_world_left"]

                if inter_mask.any():
                    root_loss = self.lambda_root * F.mse_loss(
                        j3d_left_gt[inter_mask][:, 9, :] - j3d_right_gt[inter_mask][:, 9, :],
                        j3d_left_pred[inter_mask][:, 9, :] - j3d_right_pred[inter_mask][:, 9, :]
                    )
                else:
                    root_loss = 0.0001 * F.mse_loss(
                        j3d_left_gt[:, 9, :] - j3d_right_gt[:, 9, :],
                        j3d_left_pred[:, 9, :] - j3d_right_pred[:, 9, :]
                    )
                final_loss += root_loss
                if i == num_preds - 1:
                    mano_losses[
                        "root_loss"
                    ] = root_loss.detach()

        final_loss /= num_preds
        mano_losses["mano_total_loss"] = final_loss.detach()
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
        lambda_end2end_joints2d=None,
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
        self.lambda_end2end_joints2d = lambda_end2end_joints2d
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
        if gts["verts3d"].shape[0] != batch:
            expanded_gts_verts3d = gts["verts3d"].view(batch, T, *gts["verts3d"].shape[1:])
        else:
            expanded_gts_verts3d = gts["verts3d"]
        if gts["joints3d"].shape[0] != batch:
            expanded_gts_joints3d = gts["joints3d"].view(
                batch, T, *gts["joints3d"].shape[1:]
            )
        else:
            expanded_gts_joints2d = gts["joints3d"]
        if gts["joints2d"].shape[0] != batch:
            expanded_gts_joints2d = gts["joints2d"].view(
                batch, T, *gts["joints2d"].shape[1:]
            )
        else:
            expanded_gts_joints2d = gts["joints2d"]
        if gts["mano_pose"].shape[0] != batch:
            expanded_gts_mano_pose = gts["mano_pose"].view(
                batch, T, *gts["mano_pose"].shape[1:]
            )
        else:
            expanded_gts_mano_pose = gts["mano_pose"]
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
                    ] = dynamic_mesh3d_loss.detach()
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
                    ] = dynamic_joints3d_loss.detach()
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
                    ] = dynamic_mano_pose_loss.detach()
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
                    ] = temporal_mesh3d_loss.detach()
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
                    ] = temporal_joints3d_loss.detach()
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
                    ] = temporal_manopose_loss.detach()
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
                    ] = end2end_mesh3d_loss.detach()
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
                    ] = end2end_joints3d_loss.detach()
            # end2end joints2d
            if (
                (self.lambda_end2end_joints2d is not None)
                and ("t_joints2d" in pred)
                and ("joints2d" in gts)
            ):
                joints2d_center = expanded_gts_joints2d[:, T // 2]
                mask = (joints2d_center > -0.5) * (joints2d_center < 0.5)
                mask = mask.all(dim=-1)
                if not mask.any():
                    end2end_joints2d_loss = torch.zeros_like(end2end_joints3d_loss)
                else:
                    end2end_joints2d_loss = self.lambda_end2end_joints2d * F.mse_loss(
                        pred["t_joints2d"][mask], joints2d_center[mask]
                    )
                final_loss += end2end_joints2d_loss
                if i == num_preds - 1:
                    dynamic_mano_losses[
                        "dynamic_end2end_joints2d_loss"
                    ] = end2end_joints2d_loss.detach()
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
                    ] = end2end_manopose_loss.detach()
        final_loss /= num_preds
        dynamic_mano_losses["dynamic_mano_total_loss"] = final_loss.detach()
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
