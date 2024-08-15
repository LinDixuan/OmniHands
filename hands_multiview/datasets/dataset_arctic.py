import random
import os
import smplx
import torch
import pickle
from tqdm import tqdm
import sys
from glob import glob
import collections
from typing import List, Optional, Tuple, Literal
import torchvision.transforms
from PIL import Image, ImageFilter, ImageOps
from PIL import ImageFile
from torchvision.transforms import functional
from .relighted_mano import MANO as rel_mano
from .utils import flip_pose, expand_to_aspect_ratio
from .ho3d_util import *
from .dataset_util import *

def to_homo_batch(x):
    assert isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x.shape[2] == 3
    assert len(x.shape) == 3
    batch_size = x.shape[0]
    num_pts = x.shape[1]
    x_homo = torch.ones(batch_size, num_pts, 4, device=x.device)
    x_homo[:, :, :3] = x.clone()
    return x_homo


def to_xyz_batch(x_homo):
    """
    Input: (B, N, 4)
    Ouput: (B, N, 3)
    """
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 4
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 3, device=x_homo.device)
    x = x_homo[:, :, :3] / x_homo[:, :, 3:4]
    return x


def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    x = x_homo[:, :, :2] / x_homo[:, :, 2:3]
    return x


def transform_points_batch(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (B, N, 3), in METERS!!
    world2cam_mat: (B, 4, 4)
    Output: points in cam coord (B, N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape[1:] == (4, 4)
    assert len(pts.shape) == 3
    assert pts.shape[2] == 3
    batch_size = pts.shape[0]
    pts_homo = to_homo_batch(pts)

    # mocap to cam
    pts_cam_homo = torch.bmm(world2cam_mat, pts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    pts_cam = to_xyz_batch(pts_cam_homo)

    assert pts_cam.shape[2] == 3
    return pts_cam


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d

class Arctic_Test(torch.utils.data.Dataset):
    def __init__(self, T=9, split='train'):
        # mano layer
        smplx_path = {'right': '/workspace/hamer_intertime/_DATA/data/mano/MANO_RIGHT.pkl',
                      'left': '/workspace/hamer_intertime/_DATA/data/mano/MANO_LEFT.pkl'}
        self.mano_layer = {'right': smplx.create(smplx_path['right'], 'mano', use_pca=False, is_rhand=True),
                      'left': smplx.create(smplx_path['left'], 'mano', use_pca=False, is_rhand=False)}
        self.rel_mano = rel_mano(use_pca=False)
        self.seq_len = T
        self.split = split
        self.filter_offset = 50
        self.scale_factor = 1.75
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        """if torch.sum(torch.abs(self.mano_layer['left'].shapedirs[:, 0, :] - self.mano_layer['right'].shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            self.mano_layer['left'].shapedirs[:, 0, :] *= -1"""

        root_path = '/workspace/twohand_full/arctic/arctic_data/data'
        split_json_path = '/workspace/twohand_full/arctic/arctic_data/data/splits_json/protocol_p1.json'

        with open(split_json_path, 'r') as f:
            split_seq = json.load(f)

        with open(os.path.join(root_path, 'meta', 'misc.json')) as f:
            self.cam_params = json.load(f)

        self.seq_names = []
        for nam in split_seq[split]:
            self.seq_names.append(nam)

        annot_root_path = '/workspace/arctic/outputs/processed/seqs'

        self.mano_params = {}
        self.seq_size = []
        self.seq_dir = []
        self.bbox = {}

        for seq_name in tqdm(self.seq_names):
            pid = seq_name.split('/')[-2]
            sid = seq_name.split('/')[-1]
            proc_param = np.load(os.path.join(annot_root_path, pid, f'{sid}.npy'), allow_pickle=True).item()
            mano_param = np.load(os.path.join('/workspace/twohand_full/arctic/arctic_data/data/raw_seqs',
                                                  pid, f'{sid}.mano.npy'), allow_pickle=True).item()
            bboxes = proc_param['bbox']
            self.bbox[seq_name] = bboxes
            self.mano_params[seq_name] = mano_param
            for cid in range(1, 9):
                self.seq_size.append(len(os.listdir(os.path.join(root_path, 'cropped_images', seq_name, f"{cid}"))) -
                                     self.filter_offset * 2)
                self.seq_dir.append(os.path.join(seq_name, f"{cid}"))

        self.seq_size = np.array(self.seq_size)
        self.seq_cumsum = np.cumsum(self.seq_size)
        self.data_size = self.seq_size.sum()

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        seqId = np.searchsorted(self.seq_cumsum, index, side='right')
        seq_offset = 0 if seqId == 0 else self.seq_cumsum[seqId - 1]
        idx = index - seq_offset + self.filter_offset
        seq_dir = self.seq_dir[seqId]

        pid, sid, cid = seq_dir.split('/')
        view_idx = int(cid)
        seq_name = os.path.join(pid, sid)
        vidx = idx - self.cam_params[pid]["ioi_offset"]

        bbox = torch.FloatTensor(self.bbox[seq_name][vidx, view_idx])
        bbox_loose = bbox.clone()
        bbox_loose[2] *= 1.5 * 200
        cx, cy, dim = bbox_loose
        bbox_corner = torch.FloatTensor([cx - dim / 2, cy - dim / 2])

        img_root_path = '/workspace/twohand_full/arctic/arctic_data/data/cropped_images'

        mano_params = self.mano_params[seq_name]
        fname = f"{img_root_path}/{seq_name}/{view_idx}/{idx:05d}.jpg"

        img = Image.open(fname).convert("RGB")
        cropped_size = img.size

        gt_params = {'right':{}, 'left':{}}
        for hand_type in ['right', 'left']:
            mano_param = mano_params[hand_type]
            root_pose = torch.FloatTensor(mano_param['rot'][vidx]).view(1, 3)
            hand_pose = torch.FloatTensor(mano_param['pose'][vidx]).view(1, -1)
            shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
            trans = torch.FloatTensor(mano_param['trans'][vidx]).view(1, 3)
            output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)

            verts = output.vertices

            world2cam = torch.FloatTensor(np.array(self.cam_params[pid]["world2cam"][view_idx - 1])).unsqueeze(0)
            verts_cam = transform_points_batch(world2cam, verts)
            joints = torch.FloatTensor(self.rel_mano.sh_joint_regressor).to(verts_cam) @ verts_cam
            intris_mat = torch.FloatTensor(np.array(self.cam_params[pid]["intris_mat"][view_idx - 1])).unsqueeze(0)
            verts2d = project2d_batch(intris_mat, verts_cam)[0]
            verts2d = verts2d - bbox_corner[None, :]
            verts2d = verts2d / bbox_loose[2] * torch.FloatTensor(cropped_size)[None, :]

            gt_params[hand_type]['verts2d'] = verts2d.detach().cpu().numpy().copy()
            gt_params[hand_type]['verts3d'] = verts_cam
            gt_params[hand_type]['joints3d'] = joints
            gt_params[hand_type]['mano_param'] = torch.cat([root_pose, hand_pose, shape], dim=-1)[0]
            gt_params[hand_type]['pose'] = torch.cat([root_pose, hand_pose], dim=-1)[0]
            gt_params[hand_type]['shape'] = shape[0]

        crop_hand_right = get_bbox_joints(gt_params['right']['verts2d'], bbox_factor=self.scale_factor)
        crop_hand_left = get_bbox_joints(gt_params['left']['verts2d'], bbox_factor=self.scale_factor)
        center_right = (crop_hand_right[:2] + crop_hand_right[2:]) / 2
        center_left = (crop_hand_left[:2] + crop_hand_left[2:]) / 2
        scale_right = crop_hand_right[2:] - crop_hand_right[:2]
        scale_right = expand_to_aspect_ratio(scale_right, [192, 256])

        scale_left = crop_hand_left[2:] - crop_hand_left[:2]
        scale_left = expand_to_aspect_ratio(scale_left, [192, 256])
        scale_right = scale_right.max()
        scale_left = scale_left.max()
        assert scale_right > 0, f"{center_right} {scale_right} "
        assert scale_left > 0, f"{center_left} {scale_left}"

        # scale_left = scale_right = max(scale_left, scale_right)
        affinetrans_right, _ = get_affine_transform(
            center_right, scale_right, [256, 256]
        )
        affinetrans_left, _ = get_affine_transform(
            center_left, scale_left, [256, 256]
        )

        bbox_right = np.array([*(center_right - scale_right / 2), scale_right, scale_right], dtype=np.int32)
        bbox_left = np.array([*(center_left - scale_left / 2), scale_left, scale_left], dtype=np.int32)
        right_corner = bbox_right.copy()
        right_corner[2:] = right_corner[2:] + right_corner[:2]
        left_corner = bbox_left.copy()
        left_corner[2:] = left_corner[2:] + left_corner[:2]

        v2d_right = (gt_params['right']['verts2d'] - right_corner[None, :2]) / scale_right * 2 - 1
        v2d_left = (gt_params['left']['verts2d'] - left_corner[None, :2]) / scale_left * 2 - 1

        img_right = img.crop(right_corner.tolist())
        img_right = img_right.resize((256, 256))
        img_left = img.crop(left_corner.tolist())
        img_left = img_left.resize((256, 256))
        img_full = img
        bbox_hand_right = bbox_right
        bbox_hand_left = bbox_left
        bbox_full = np.zeros_like(bbox_hand_right)
        bbox_full[2], bbox_full[3] = 256, 256

        img_right = functional.to_tensor(img_right).permute(1, 2, 0).flip(2) * 255
        img_left = functional.to_tensor(img_left).permute(1, 2, 0).flip(2) * 255
        img_full = functional.to_tensor(img_full).permute(1, 2, 0).flip(2) * 255

        for n_c in range(3):
            img_right[:, :, n_c] = (img_right[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
            img_left[:, :, n_c] = (img_left[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]

        # -------flip-------
        pose_left = gt_params['left']['mano_param'][:48].reshape(16, 3)
        pose_left[:, 1:3] = -pose_left[:, 1:3]
        gt_params['left']['mano_param'][:48] = pose_left.reshape(48)
        img_left = img_left.flip(1)
        # ------------------
        sample = {}

        sample['img_full'] = img_full
        sample['img_right'] = img_right
        sample['img_left'] = img_left
        sample['bbox_right'] = bbox_hand_right
        sample['bbox_left'] = bbox_hand_left
        sample['bbox_full'] = bbox_full
        sample['verts3d_right'] = gt_params['right']['verts3d'][0] - gt_params['right']['joints3d'][0][0:1]
        sample['verts3d_left'] = gt_params['left']['verts3d'][0] - gt_params['right']['joints3d'][0][0:1]
        sample['verts2d_full_right'] = gt_params['right']['verts2d']
        sample['verts2d_full_left'] = gt_params['left']['verts2d']

        sample['verts2d_right'] = v2d_right
        sample['verts2d_left'] = v2d_left
        sample['joints3d_right'] = gt_params['right']['joints3d'][0] - gt_params['right']['joints3d'][0][0:1]
        sample['joints3d_left'] = gt_params['left']['joints3d'][0] - gt_params['right']['joints3d'][0][0:1]
        sample['mano_param_right'] = gt_params['right']['mano_param']
        sample['mano_param_left'] = gt_params['left']['mano_param']
        sample['mano_pose_right'] = gt_params['right']['pose']
        sample['mano_pose_left'] = gt_params['left']['pose']
        sample['mano_shape_right'] = gt_params['right']['shape']
        sample['mano_shape_left'] = gt_params['left']['shape']

        for k in sample.keys():
            sample[k] = torch.FloatTensor(sample[k])
            sample[k] = torch.stack([sample[k]] * self.seq_len, dim=0)

        return sample