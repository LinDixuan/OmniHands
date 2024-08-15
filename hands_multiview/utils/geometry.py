from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
import einops
from plyfile import PlyData, PlyElement

def get_close_faces():
    file_path = '/workspace/hamer_intertime/_DATA/left_close.ply'
    plydata = PlyData.read(file_path)
    faces_l_close = plydata.elements[1].data
    fc = []

    for face_id in range(faces_l_close.shape[0]):
        fc.append(faces_l_close[face_id][0])

    flc = np.stack(fc, axis=0)

    file_path = '/workspace/hamer_intertime/_DATA/right_close.ply'
    plydata = PlyData.read(file_path)
    faces_l_close = plydata.elements[1].data
    fc = []

    for face_id in range(faces_l_close.shape[0]):
        fc.append(faces_l_close[face_id][0])

    frc = np.stack(fc, axis=0)

    return frc, flc

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def distance_map(bboxes, flip=False):       # bs x 2 x 4
    bs = bboxes.shape[0]
    bbox_1 = bboxes[:,0,:].clone()
    bbox_2 = bboxes[:,1,:].clone()

    if flip:
        x_cen_1 = bbox_1[:, 0] + bbox_1[:, 2] / 2
        x_cen_2 = bbox_2[:, 0] + bbox_2[:, 2] / 2
        x_cen_1_flip = x_cen_2 - (x_cen_1 - x_cen_2)
        bbox_1[:, 0] = x_cen_1_flip - bbox_1[:, 2] / 2

    corner_1 = bbox_1.clone()
    corner_1[:, 2:] = corner_1[:, 2:] + corner_1[:, :2]
    corner_2 = bbox_2.clone()
    corner_2[:, 2:] = corner_2[:, 2:] + corner_2[:, :2]

    cen_1 = torch.cat([bbox_1[:, 0:1] + bbox_1[:, 2:3] / 2, bbox_1[:, 1:2] + bbox_1[:, 3:4] / 2], dim=-1)   # bs x 2
    cen_2 = torch.cat([bbox_2[:, 0:1] + bbox_2[:, 2:3] / 2, bbox_2[:, 1:2] + bbox_2[:, 3:4] / 2], dim=-1)

    def generate_matrix(size, bs):
        x_values = torch.arange(0, size).unsqueeze(0).expand(size, -1) / size
        y_values = torch.arange(0, size).unsqueeze(1).expand(-1, size) / size
        return torch.stack([x_values, y_values], dim=2).unsqueeze(0).repeat(bs,1,1,1)
    map_1 = generate_matrix(16,bs).to(bboxes)   # bs x 16 x 16 x 2
    map_2 = generate_matrix(16,bs).to(bboxes)   # bs x 16 x 16 x 2
    map_1[..., 0] = map_1[..., 0] * bbox_1[..., 2].unsqueeze(1).unsqueeze(1) + bbox_1[..., 0].unsqueeze(1).unsqueeze(1)
    map_1[..., 1] = map_1[..., 1] * bbox_1[..., 3].unsqueeze(1).unsqueeze(1) + bbox_1[..., 1].unsqueeze(1).unsqueeze(1)
    map_2[..., 0] = map_2[..., 0] * bbox_2[..., 2].unsqueeze(1).unsqueeze(1) + bbox_2[..., 0].unsqueeze(1).unsqueeze(1)
    map_2[..., 1] = map_2[..., 1] * bbox_2[..., 3].unsqueeze(1).unsqueeze(1) + bbox_2[..., 1].unsqueeze(1).unsqueeze(1)

    map_1to2 = (map_1 - map_2) / bbox_1[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    map_2to1 = (map_2 - map_1) / bbox_2[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    map_1to2_cen = (map_1 - cen_2.unsqueeze(1).unsqueeze(1)) / bbox_1[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    map_2to1_cen = (map_2 - cen_1.unsqueeze(1).unsqueeze(1)) / bbox_2[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    map_1to1_cen = (map_1 - cen_1.unsqueeze(1).unsqueeze(1)) / bbox_1[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    map_2to2_cen = (map_2 - cen_2.unsqueeze(1).unsqueeze(1)) / bbox_2[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    IoU_1 = torch.ones_like(map_1[..., 0:1]) * 0.01
    IoU_2 = torch.ones_like(map_2[..., 0:1]) * 0.01

    # bs x 16 x 16
    mask_1 = (map_1[:, :, :, 0] + bbox_1[..., 2].unsqueeze(1).unsqueeze(1) / 16 > corner_1[:, None, None, 0]) * \
             (map_1[:, :, :, 1] + bbox_1[..., 3].unsqueeze(1).unsqueeze(1) / 16 > corner_1[:, None, None, 1]) * \
             (map_1[:, :, :, 0] < corner_1[:, None, None, 2]) * \
             (map_1[:, :, :, 1] < corner_1[:, None, None, 3])

    mask_2 = (map_2[:, :, :, 0] + bbox_2[..., 2].unsqueeze(1).unsqueeze(1) / 16 > corner_2[:, None, None, 0]) * \
             (map_2[:, :, :, 1] + bbox_2[..., 3].unsqueeze(1).unsqueeze(1) / 16 > corner_2[:, None, None, 1]) * \
             (map_2[:, :, :, 0] < corner_2[:, None, None, 2]) * \
             (map_2[:, :, :, 1] < corner_2[:, None, None, 3])

    IoU_1[~mask_1] = -0.01
    IoU_2[~mask_2] = -0.01

    return map_1to2, map_2to1, map_1to2_cen, map_2to1_cen, map_1to1_cen, map_2to2_cen, IoU_1, IoU_2


def add_map(box_right, box_left, activate=False):
    """map_L2M, map_M2L, map_L2M_cen, map_M2L_cen, map_L2L_cen, _, _, _ = \
        distance_map(torch.stack([box_left, box_full], dim=1), flip=True)
    map_R2M, map_M2R, map_R2M_cen, map_M2R_cen, map_R2R_cen, _, _, _ = \
        distance_map(torch.stack([box_right, box_full], dim=1), flip=False)"""
    map_L2R_f, map_R2L_f, map_L2R_cen_f, map_R2L_cen_f, _, _, IoU_L_f, IoU_R_f = \
        distance_map(torch.stack([box_right, box_left], dim=1), flip=True)
    map_R2L, map_L2R, map_R2L_cen, map_L2R_cen, _, _, IoU_R, IoU_L = \
        distance_map(torch.stack([box_right, box_left], dim=1), flip=False)

    if activate:
        map_L2R = torch.sigmoid(map_L2R)
        map_R2L = torch.sigmoid(map_R2L)
        map_L2R_f = torch.sigmoid(map_L2R_f)
        map_R2L_f = torch.sigmoid(map_R2L_f)
        map_R2L_cen = torch.sigmoid(map_R2L_cen)
        map_L2R_cen = torch.sigmoid(map_L2R_cen)
        map_L2R_cen_f = torch.sigmoid(map_L2R_cen_f)
        map_R2L_cen_f = torch.sigmoid(map_R2L_cen_f)

    side_L = torch.ones_like(map_R2L[..., 0:1]) * -0.01
    side_R = torch.ones_like(map_R2L[..., 0:1]) * 0.01
    map_L = torch.cat([map_L2R_f, map_R2L_f, map_L2R_cen_f, map_R2L_cen_f, IoU_L_f, IoU_R_f, side_L], dim=-1)  # B x 16 x 16 x 11
    map_R = torch.cat([map_R2L, map_L2R, map_R2L_cen, map_L2R_cen, IoU_R, IoU_L, side_R], dim=-1)

    pos_R, pos_L = map_R[:, :, 2:-2, :], map_L[:, :, 2:-2, :]   # B x 16 x 12 x 11

    return pos_R, pos_L

def combine_box(bboxes): #bs x 2 x 4
    bs = bboxes.shape[0]
    if isinstance(bboxes,torch.Tensor):
        bboxes = bboxes.numpy().copy()
    bboxes[:,:,2:] = bboxes[:,:,2:] + bboxes[:,:,:2]
    left_top = bboxes[:,:,:2].min(axis=1)
    right_down = bboxes[:,:,2:].max(axis=1) - left_top

    comb = np.concatenate([left_top,right_down],axis=-1)
    return comb

def merge_bbox(bbox_list):
    if isinstance(bbox_list,torch.Tensor):
        bbox_list = bbox_list.numpy()
    bbox = np.stack(bbox_list, axis=0)  # N x ... x 4
    x_min = np.min(bbox[..., 0], axis=0)
    y_min = np.min(bbox[..., 1], axis=0)
    x_max = np.max(bbox[..., 0] + bbox[..., 2], axis=0)
    y_max = np.max(bbox[..., 1] + bbox[..., 3], axis=0)
    bbox = np.stack([x_min, y_min, x_max - x_min, y_max - y_min], axis=-1).astype(np.int32)
    return bbox

def bbox_2_square(bbox, ratio=1.0):

    x_min = bbox[..., 0]
    y_min = bbox[..., 1]
    x_max = bbox[..., 0] + bbox[..., 2]
    y_max = bbox[..., 1] + bbox[..., 3]

    mid = np.stack([(x_min + x_max) / 2, (y_min + y_max) / 2], axis=-1)  # ... x 2
    L = np.max(np.stack([(x_max - x_min), (y_max - y_min)], axis=-1), axis=-1)[..., np.newaxis]  # ... x 1
    L = L / ratio

    bbox = np.concatenate([mid - L / 2, L, L], axis=-1).astype(np.int32)
    return bbox

def proj_orth(v3d, scale, uv):
    if scale.dim() == uv.dim() - 1:
        scale = scale.unsqueeze(1)
    v2d = 2 * v3d[..., :2] * scale.unsqueeze(1) + uv.unsqueeze(1)
    #v2d = v3d[..., :2] * scale.unsqueeze(1) + uv.unsqueeze(1)
    return v2d