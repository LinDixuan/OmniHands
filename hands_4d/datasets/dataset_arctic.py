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

class Arctic_Temp(torch.utils.data.Dataset):
    def __init__(self, seq_len=9, gap=10, split='train'):
        # mano layer
        layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False,
                     'create_transl': False, 'num_pca_comps': 45, 'flat_hand_mean': False}
        human_model_path = '/workspace/hamer_twohand/_DATA/data/'
        self.mano_layer = {'right': smplx.create(human_model_path, 'mano', use_pca=False, is_rhand=True, **layer_arg),
                      'left': smplx.create(human_model_path, 'mano', use_pca=False, is_rhand=False, **layer_arg)}
        self.rel_mano = rel_mano(use_pca=False)
        self.seq_len = seq_len
        self.gap = gap

        self.split = split
        self.filter_offset = 100
        self.scale_factor = 1.75
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)

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
        self.bbox = {}

        for seq_name in self.seq_names:
            pid = seq_name.split('/')[-2]
            sid = seq_name.split('/')[-1]
            proc_param = np.load(os.path.join(annot_root_path, pid, f'{sid}.npy'), allow_pickle=True).item()
            mano_param = np.load(os.path.join('/workspace/twohand_full/arctic/arctic_data/data/raw_seqs',
                                                  pid, f'{sid}.mano.npy'), allow_pickle=True).item()
            bboxes = proc_param['bbox']
            self.bbox[seq_name] = bboxes
            self.mano_params[seq_name] = mano_param

            self.seq_size.append(len(os.listdir(os.path.join(root_path, 'cropped_images', seq_name, f"1"))) -
                                 self.filter_offset * 2)

        self.seq_size = np.array(self.seq_size)
        self.seq_cumsum = np.cumsum(self.seq_size)
        self.data_size = self.seq_size.sum()

    def mano_transform(self, mano_pose, mano_shape, hand_type='right'):
        # mano_pose: axis    [bs x 48]  flat_mean = False
        global_orient = mano_pose[:, :3]
        hand_pose = mano_pose[:, 3:48]
        if self.rel_mano.layer[hand_type].pose_mean.device != hand_pose.device:
            self.rel_mano.layer[hand_type] = self.rel_mano.layer[hand_type].to(hand_pose.device)
        outputs = self.rel_mano.layer[hand_type](betas=mano_shape,
                                                   hand_pose=hand_pose, global_orient=global_orient)

        verts = outputs.vertices
        joints = torch.FloatTensor(self.rel_mano.sh_joint_regressor).to(verts) @ verts
        return verts, joints

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        seqId = np.searchsorted(self.seq_cumsum, index, side='right')
        seq_offset = 0 if seqId == 0 else self.seq_cumsum[seqId - 1]
        idx = index - seq_offset + self.filter_offset
        seq_name = self.seq_names[seqId]
        img_root_path = '/workspace/twohand_full/arctic/arctic_data/data/cropped_images'

        view_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        seq_ids = [idx + self.gap * (ind - self.seq_len // 2 - 3) for ind in range(self.seq_len + 5)]

        random.shuffle(view_ids)
        view_idx = view_ids[0]
        pid, sid = seq_name.split('/')

        mano_params = self.mano_params[seq_name]
        outputs = collections.defaultdict(list)

        for id in seq_ids:
            vidx = id - self.cam_params[pid]["ioi_offset"]
            fname = f"{img_root_path}/{seq_name}/{view_idx}/{id:05d}.jpg"
            if not os.path.exists(fname):
                continue
            bbox = torch.FloatTensor(self.bbox[seq_name][vidx, view_idx])
            bbox_loose = bbox.clone()
            bbox_loose[2] *= 1.5 * 200
            cx, cy, dim = bbox_loose
            bbox_corner = torch.FloatTensor([cx - dim / 2, cy - dim / 2])
            try:
                img = Image.open(fname).convert("RGB")
            except:
                continue
            cropped_size = img.size
            gt_params = {'right':{}, 'left':{}}

            verts_world = {}
            hand_pose = {}
            root_pose = {}
            shapes = {}

            for hand_type in ['right', 'left']:
                mano_param = mano_params[hand_type]
                root_pose[hand_type] = torch.FloatTensor(mano_param['rot'][vidx]).view(1, 3)
                hand_pose[hand_type] = torch.FloatTensor(mano_param['pose'][vidx]).view(1, -1)
                shapes[hand_type] = torch.FloatTensor(mano_param['shape']).view(1, -1)
                trans = torch.FloatTensor(mano_param['trans'][vidx]).view(1, 3)
                output = self.mano_layer[hand_type](global_orient=root_pose[hand_type].clone(),
                                                    hand_pose=hand_pose[hand_type].clone(),
                                                    betas=shapes[hand_type].clone(),
                                                    transl=trans)
                verts_world[hand_type] = output.vertices

                world2cam = torch.FloatTensor(np.array(self.cam_params[pid]["world2cam"][view_idx - 1])).unsqueeze(0)
                cam_R = world2cam[0, :3, :3]

                verts_cam = transform_points_batch(world2cam, verts_world[hand_type])
                joints = torch.FloatTensor(self.rel_mano.sh_joint_regressor).to(verts_cam) @ verts_cam
                intris_mat = torch.FloatTensor(np.array(self.cam_params[pid]["intris_mat"][view_idx - 1])).unsqueeze(0)
                verts2d = project2d_batch(intris_mat, verts_cam)[0]
                verts2d = verts2d - bbox_corner[None, :]
                verts2d = verts2d / bbox_loose[2] * torch.FloatTensor(cropped_size)[None, :]

                joints2d = project2d_batch(intris_mat, joints)[0]
                joints2d = joints2d - bbox_corner[None, :]
                joints2d = joints2d / bbox_loose[2] * torch.FloatTensor(cropped_size)[None, :]

                root_pose_R, _ = cv2.Rodrigues(root_pose[hand_type].numpy()[0])
                new_R = np.dot(cam_R, root_pose_R)
                root_pose_R, _ = cv2.Rodrigues(new_R)
                root_pose_R = torch.FloatTensor(root_pose_R).view(1, 3)

                gt_params[hand_type]['verts2d'] = verts2d.detach().cpu().numpy().copy()
                gt_params[hand_type]['joints2d'] = joints2d.detach().cpu().numpy().copy()
                gt_params[hand_type]['verts3d'] = verts_cam
                gt_params[hand_type]['joints3d'] = joints
                gt_params[hand_type]['mano_param'] = torch.cat([root_pose_R.clone(),
                                                                hand_pose[hand_type].clone(),
                                                                shapes[hand_type].clone()], dim=-1)[0]
                gt_params[hand_type]['pose'] = torch.cat([root_pose_R.clone(), hand_pose[hand_type].clone()], dim=-1)[0]
                gt_params[hand_type]['shape'] = shapes[hand_type][0]

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

            if self.split == 'train':
                center_rand = np.random.uniform(low=-1, high=1, size=2)
                scale_rand = np.random.randn()

                # Scale jittering
                scale_jittering = 0.2 * scale_rand + 1
                scale_jittering = np.clip(
                    scale_jittering, 1 - 0.2, 1 + 0.2
                )
                scale_right = scale_right * scale_jittering
                scale_left = scale_left * scale_jittering

                # Randomly jitter center
                center_offsets_right = 0.2 * scale_right * center_rand
                center_offsets_left = 0.2 * scale_left * center_rand
                center_right = center_right + center_offsets_right
                center_left = center_left + center_offsets_left

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

            v2d_right = (gt_params['right']['verts2d'] - right_corner[None, :2]) / scale_right - 0.5
            v2d_left = (gt_params['left']['verts2d'] - left_corner[None, :2]) / scale_left - 0.5
            j2d_right = (gt_params['right']['joints2d'] - right_corner[None, :2]) / scale_right - 0.5
            j2d_left = (gt_params['left']['joints2d'] - left_corner[None, :2]) / scale_left - 0.5

            img_right = img.crop(right_corner.tolist())
            img_right = img_right.resize((256, 256))
            img_left = img.crop(left_corner.tolist())
            img_left = img_left.resize((256, 256))
            img_full = img.resize((256, 256))
            bbox_hand_right = bbox_right
            bbox_hand_left = bbox_left
            bbox_full = np.zeros_like(bbox_hand_right)
            bbox_full[2], bbox_full[3] = cropped_size[0], cropped_size[1]

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
            j2d_left[:, 0] *= -1
            v2d_left[:, 0] *= -1
            # ------------------
            sample = {}
            sample['img_full'] = img_full
            sample['img_right'] = img_right
            sample['img_left'] = img_left
            sample['bbox_right'] = bbox_hand_right
            sample['bbox_left'] = bbox_hand_left
            sample['bbox_full'] = bbox_full
            sample['verts3d_world_right'] = gt_params['right']['verts3d'][0]
            sample['verts3d_world_left'] = gt_params['left']['verts3d'][0]
            sample['joints2d_right'] = j2d_right
            sample['joints2d_left'] = j2d_left
            sample['joints3d_world_right'] = gt_params['right']['joints3d'][0]
            sample['joints3d_world_left'] = gt_params['left']['joints3d'][0]
            sample['mano_param_right'] = gt_params['right']['mano_param']
            sample['mano_param_left'] = gt_params['left']['mano_param']
            sample['mano_pose_right'] = gt_params['right']['pose']
            sample['mano_pose_left'] = gt_params['left']['pose']
            sample['mano_shape_right'] = gt_params['right']['shape']
            sample['mano_shape_left'] = gt_params['left']['shape']

            for k in sample.keys():
                outputs[k].append(torch.FloatTensor(sample[k]))
            if len(outputs['img_right']) >= self.seq_len:
                break

        for k in outputs.keys():
            outputs[k] = torch.stack(outputs[k], dim=0)
        outputs["inter"] = torch.ones_like(outputs['bbox_right'][:, 0]).float()

        return outputs

    def calc_vdif(self, gt_mano_param_right, verts_w_right, hand_type):
        #joints_w_right = torch.FloatTensor(self.rel_mano.sh_joint_regressor).to(verts_w_right) @ verts_w_right
        gt_verts3d, gt_joints3d = self.mano_transform(gt_mano_param_right[:, :48],
                                                      gt_mano_param_right[:, 48:], hand_type)
        #gt_verts3d = gt_verts3d - gt_joints3d[:, 0:1]
        #verts_w_right = verts_w_right - joints_w_right[:, 0:1]
        print(f"{hand_type} {torch.mean(gt_verts3d - verts_w_right).item() * 1000}mm")


