import random

import torch
import os
import pickle
import json
import numpy as np
import cv2 as cv
import tqdm
import collections
from PIL import Image, ImageFilter, ImageOps
from PIL import ImageFile
from torchvision.transforms import functional
from typing import List, Optional, Tuple, Literal
from ..models.modules.manolayer import ManoLayer
from .relighted_mano import get_mano_data_intertemp, MANO, cam2pixel
from .utils import flip_pose, expand_to_aspect_ratio
from .ho3d_util import *
from .dataset_util import *


class InterHandSingleFrameDataset(torch.utils.data.Dataset):
    def __init__(self,
                 label_path: str,
                 camera_path: Optional[str] = None,
                 image_path: Optional[str] = None,
                 capture_list: Optional[List[int]] = None,
                 need_image: bool = True,
                 need_camera: bool = False,
                 split='train',
                 fps='5fps',
                 hand_type='interacting',
                 T_len=7) -> None:
        super().__init__()
        self.type_list = hand_type
        self.need_image = need_image
        self.need_camera = need_camera
        self.fps = fps
        self.T = T_len
        self.split = split
        self.aug = True

        self.mano = {'left': ManoLayer('_DATA/data/mano/MANO_LEFT.pkl'),
                         'right': ManoLayer('_DATA/data/mano/MANO_RIGHT.pkl')}
        self.smplx_mano = MANO()
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)

        if capture_list is None or len(capture_list) == 0:
            capture_list = os.listdir(label_path)

            for i in range(len(capture_list)):
                capture_list[i] = int(capture_list[i].lstrip('Capture'))
        capture_list.sort()

        if split == 'train':
            seq_path = '/workspace/twohand_crop/seq_lists.json'
        elif split == 'val':
            seq_path = '/workspace/twohand_crop/seq_lists_val.json'
        else:
            seq_path = '/workspace/twohand_crop/seq_list_test.json'

        self.img_size = 256
        self.scale_factor = 2.0
        if need_camera:
            with open(camera_path, 'r') as file:
                self.cam_anno = json.load(file)
        self.label_path = label_path
        self.image_path = image_path

        self.seq_data_list = []
        self.seq_size = []
        with open(os.path.join(self.image_path, seq_path), 'r') as f:
            seq_lists = json.load(f)
        for capture in capture_list:
            seq_list = seq_lists[f"{capture}"]
            for seq_name in seq_list:
                seq_name = seq_name[:seq_name.find('.pkl')]
                with open(os.path.join(label_path, 'Capture' + str(capture), seq_name + '.pkl'), 'rb') as file:
                    data = pickle.load(file)

                if data['hand_type'] not in self.type_list:
                    continue
                valid_frame = np.array(data['frame_valid'])

                valid = np.array(data['frame_valid'])

                if self.fps == '5fps':
                    valid_5fps = np.array(data['5fps_valid'])
                    valid = valid * valid_frame * valid_5fps
                else:
                    valid = valid * valid_frame

                valid_per_cam = np.sum(valid, axis=0, keepdims=True) >= self.T
                valid = valid * valid_per_cam
                camId, frameId = np.where(valid > 0)

                if np.unique(camId).shape[0] < 5:
                    continue
                size = camId.shape[0]
                data['valid_cameraId_list'] = camId
                data['valid_frameId_list'] = frameId
                data['valid_map'] = valid

                needed_data_keys = ['captureId', 'seq_name', 'valid_cameraId_list', 'valid_frameId_list', 'hand_type',
                                    'camera_name_list', 'frame_name_list',
                                    'left_square_bbox', 'right_square_bbox', 'inter_square_bbox',
                                    'left_mano_params', 'right_mano_params']
                data2 = {}
                for k in needed_data_keys:
                    if k in data:
                        data2[k] = data[k]
                self.seq_data_list.append(data2)

                self.seq_size.append(size)
        self.seq_size = np.array(self.seq_size)
        self.seq_len = np.cumsum(self.seq_size)
        self.data_size = self.seq_size.sum()

    def load_camera(self, captureId: int, cam_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam_anno = self.cam_anno[str(captureId)]
        campos = cam_anno['campos'][cam_name].copy()
        camrot = cam_anno['camrot'][cam_name].copy()
        focal = cam_anno['focal'][cam_name].copy()
        princpt = cam_anno['princpt'][cam_name].copy()

        cam_t = np.array(campos, dtype=np.float32).reshape(3)
        cam_R = np.array(camrot, dtype=np.float32).reshape(3, 3)
        cam_t = -np.dot(cam_R,cam_t.reshape(3,1)).reshape(3) / 1000
        cam_k = np.array([[focal[0], 0, princpt[0]],
                          [0, focal[1], princpt[1]],
                          [0, 0, 1]], dtype=np.float32)

        return cam_k, cam_R, cam_t


    def cut_camera_k(self, K: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        camK = K.copy()
        camK[0, 2] = camK[0, 2] - bbox[0]
        camK[1, 2] = camK[1, 2] - bbox[1]
        camK[:2] = camK[:2] * self.img_size / bbox[2]
        return camK


    def __len__(self):
        if self.split == 'train':
            return self.data_size
        else:
            return self.data_size

    def getitem(self, frameIdx, camIdx, data):
        captureId = data['captureId']
        seq_name = data['seq_name']

        hand_type = data['hand_type']
        hand_type_tail = 'inter' if hand_type == 'interacting' else hand_type

        output = {}

        camera_name = data['camera_name_list'][camIdx]
        img_path = os.path.join(self.image_path, 'Capture' + str(captureId), seq_name, 'cam' + camera_name,
                                'image' + data['frame_name_list'][frameIdx] + '_' + hand_type_tail + '.jpg')

        img = Image.open(img_path).convert("RGB")

        if self.need_camera:
            cam_k, cam_R, cam_t = data['cam_k'], data['cam_R'], data['cam_t']

            cam_k = self.cut_camera_k(cam_k, data[hand_type_tail + '_square_bbox'][camIdx, frameIdx].copy())
            cam_R = torch.from_numpy(cam_R).float()
            cam_t = torch.from_numpy(cam_t).float()
            output['cam_K'] = torch.from_numpy(cam_k).float()
        else:
            cam_k, cam_R, cam_t = -1, -1, -1

        if 'left_mano_params' in data:
            root_left = torch.from_numpy(data['left_mano_params']['R'][frameIdx]).clone().float().reshape(1, 3, 3)

            output['R_left'] = Rmat_to_axis(root_left).reshape(3)   # 1 x 3
            pose_left = torch.from_numpy(data['left_mano_params']['pose'][frameIdx]).clone().float().reshape(1, 45)

            pose_left = self.mano['left'].pca2axis(pose_left) - self.mano['left'].hands_mean    # 1 x 45
            output['pose_left'] = pose_left.reshape(45)

            output['T_left'] = torch.from_numpy(data['left_mano_params']['T'][frameIdx]).clone().float().reshape(3)
            output['shape_left'] = torch.from_numpy(
                data['left_mano_params']['shape'][frameIdx]).clone().float().reshape(10)
            output['left_existence'] = torch.tensor(True)

        else:
            output['R_left'] = torch.eye(3).float()
            output['T_left'] = torch.tensor([0, 0, -999.9]).float()
            output['pose_left'] = torch.zeros(45).float()
            output['shape_left'] = torch.zeros(10).float()
            output['left_existence'] = torch.tensor(False)

        if 'right_mano_params' in data:
            root_right = torch.from_numpy(data['right_mano_params']['R'][frameIdx]).clone().float().reshape(1, 3, 3)
            output['R_right'] = Rmat_to_axis(root_right).reshape(3)
            pose_right = torch.from_numpy(data['right_mano_params']['pose'][frameIdx]).clone().float().reshape(1, 45)
            pose_right = self.mano['right'].pca2axis(pose_right) - self.mano['right'].hands_mean  # 1 x 45
            output['pose_right'] = pose_right.reshape(45)

            output['T_right'] = torch.from_numpy(data['right_mano_params']['T'][frameIdx]).clone().float().reshape(3)
            output['shape_right'] = torch.from_numpy(
                data['right_mano_params']['shape'][frameIdx]).clone().float().reshape(10)
            output['right_existence'] = torch.tensor(True)

        else:
            output['R_right'] = torch.eye(3).float()
            output['T_right'] = torch.tensor([0, 0, -999.9]).float()
            output['pose_right'] = torch.zeros(45).float()
            output['shape_right'] = torch.zeros(10).float()
            output['right_existence'] = torch.tensor(False)

        output['cam_R'], output['cam_T'] = cam_R, cam_t
        focal = torch.FloatTensor([cam_k[0,0],cam_k[1,1]])
        princpt = torch.FloatTensor([cam_k[0,2],cam_k[1,2]])


        manodata_left = {'pose': torch.cat([output['R_left'], output['pose_left']], dim=-1),
                         'shape': output['shape_left'],
                         'trans': output['T_left'], 'hand_type': 'left'}
        cam_left = {'R': output['cam_R'].numpy(), 't': output['cam_T'].numpy(),
                    'focal': focal.numpy(), 'princpt': princpt.numpy()}

        manodata_right = {'pose': torch.cat([output['R_right'], output['pose_right']], dim=-1),
                          'shape': output['shape_right'],
                          'trans': output['T_right'], 'hand_type': 'right'}
        cam_right = {'R': output['cam_R'].numpy(), 't': output['cam_T'].numpy(),
                     'focal': focal.numpy(), 'princpt': princpt.numpy()}


        joint_img, joint_cam, mesh_img, mesh_cam, pose, shape = get_mano_data_intertemp(manodata_left, cam_left, False,
                                                                                  (256, 256), self.smplx_mano)
        joints2d_left = joint_img
        joints3d_left = joint_cam
        verts3d_left = mesh_cam
        mano_param_left = torch.cat([pose,shape],dim=0)

        joint_img, joint_cam, mesh_img, mesh_cam, pose, shape = get_mano_data_intertemp(manodata_right, cam_right, False,
                                                                                  (256, 256), self.smplx_mano)
        joints2d_right = joint_img
        joints3d_right = joint_cam
        verts3d_right = mesh_cam
        mano_param_right = torch.cat([pose, shape], dim=0)


        crop_hand_right = get_bbox_joints(joints2d_right, bbox_factor=self.scale_factor)
        crop_hand_left = get_bbox_joints(joints2d_left, bbox_factor=self.scale_factor)
        center_right = (crop_hand_right[:2] + crop_hand_right[2:]) / 2
        center_left = (crop_hand_left[:2] + crop_hand_left[2:]) / 2
        scale_right = crop_hand_right[2:] - crop_hand_right[:2]
        scale_right = expand_to_aspect_ratio(scale_right, [192, 256])
        scale_left = crop_hand_left[2:] - crop_hand_left[:2]
        scale_left = expand_to_aspect_ratio(scale_left, [192, 256])
        scale_right = scale_right.max()
        scale_left = scale_left.max()

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

        joints2d_right = transform_coords(
            joints2d_right, affinetrans_right
        )  # hand landmark trans
        joints2d_left = transform_coords(
            joints2d_left, affinetrans_left
        )  # hand landmark trans

        joints2d_right = joints2d_right / 256 - 0.5
        joints2d_left = joints2d_left / 256 - 0.5

        bbox_right = np.array([*(center_right - scale_right / 2), scale_right, scale_right], dtype=np.int32)
        bbox_left = np.array([*(center_left - scale_left / 2), scale_left, scale_left], dtype=np.int32)
        right_corner = bbox_right.copy()
        right_corner[2:] = right_corner[2:] + right_corner[:2]
        left_corner = bbox_left.copy()
        left_corner[2:] = left_corner[2:] + left_corner[:2]
        img_right = img.crop(right_corner.tolist())
        img_right = img_right.resize((256, 256))
        img_left = img.crop(left_corner.tolist())
        img_left = img_left.resize((256, 256))
        img_full = img.resize((256, 256))
        bbox_hand_right = bbox_right
        bbox_hand_left = bbox_left
        bbox_full = np.zeros_like(bbox_hand_right)
        bbox_full[2], bbox_full[3] = 256, 256

        img_right = functional.to_tensor(img_right).permute(1,2,0).flip(2) * 255
        img_left = functional.to_tensor(img_left).permute(1,2,0).flip(2) * 255
        img_full = functional.to_tensor(img_full).permute(1,2,0).flip(2) * 255

        for n_c in range(3):
            img_right[:, :, n_c] = (img_right[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
            img_left[:, :, n_c] = (img_left[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]

        #-------flip-------
        pose_left = mano_param_left[:48].reshape(16, 3)
        pose_left[:, 1:3] = -pose_left[:, 1:3]
        mano_param_left[:48] = pose_left.reshape(48)
        img_left = img_left.flip(1)
        joints2d_left[:, 0] = - joints2d_left[:, 0]
        #------------------
        sample = {}

        sample['img_full'] = img_full
        sample['img_right'] = img_right
        sample['img_left'] = img_left
        sample['joints2d_right'] = joints2d_right
        sample['joints2d_left'] = joints2d_left
        sample['joints3d_right'] = joints3d_right
        sample['joints3d_left'] = joints3d_left
        sample['verts3d_right'] = verts3d_right
        sample['verts3d_left'] = verts3d_left
        sample['bbox_hand_right'] = bbox_hand_right
        sample['bbox_hand_left'] = bbox_hand_left
        sample['bbox_full'] = bbox_full
        sample['mano_param_right'] = mano_param_right
        sample['mano_param_left'] = mano_param_left

        for k in sample.keys():
            sample[k] = torch.FloatTensor(sample[k])

        return sample


    def __getitem__(self, i):
        seqId = np.searchsorted(self.seq_len, i, side='right')
        seq_offset = 0 if seqId == 0 else self.seq_len[seqId - 1]
        idx = i - seq_offset

        data = self.seq_data_list[seqId].copy()
        frameId = data['valid_frameId_list'][idx]
        camIds = np.unique(data['valid_cameraId_list'])
        captureId = data['captureId']
        seq_name = data['seq_name']
        np.random.shuffle(camIds)
        sample = collections.defaultdict(list)

        for camIdx in camIds.tolist():
            camera_name = data['camera_name_list'][camIdx]
            img_path = os.path.join(self.image_path, 'Capture' + str(captureId), seq_name, 'cam' + camera_name,
                                    'image' + data['frame_name_list'][frameId] + '_' + 'inter' + '.jpg')
            if not os.path.exists(img_path):
                continue
            cam_k, cam_R, cam_t = self.load_camera(captureId, camera_name)
            data['cam_k'], data['cam_R'], data['cam_t'] = cam_k, cam_R, cam_t
            output = self.getitem(frameId, camIdx, data)
            for handtype in ['left', 'right']:
                # for k in output.keys():
                #    sample[k].append(output[k])
                sample[f"img_{handtype}"].append(output[f"img_{handtype}"])
                sample[f"bbox_{handtype}"].append(output[f"bbox_hand_{handtype}"])
                sample[f"joints2d_{handtype}"].append(output[f"joints2d_{handtype}"])
                sample[f"joints3d_world_{handtype}"].append(output[f"joints3d_{handtype}"])
                sample[f"verts3d_world_{handtype}"].append(output[f"verts3d_{handtype}"])
                mano_param = output[f"mano_param_{handtype}"]
                sample[f"mano_param_{handtype}"].append(mano_param)
                sample[f"mano_pose_{handtype}"].append(mano_param[:48])
                sample[f"mano_shape_{handtype}"].append(mano_param[48:])

            sample["img_full"].append(output["img_full"])
            sample["bbox_full"].append(output["bbox_full"])
            if len(sample["img_full"]) >= self.T:
                break

        for k in sample.keys():
            if isinstance(sample[k][0], np.ndarray):
                sample[k] = torch.from_numpy(np.stack(sample[k], axis=0))
            elif isinstance(sample[k][0], torch.Tensor):
                sample[k] = torch.stack(sample[k],dim=0)
            else:
                sample[k] = torch.FloatTensor(sample[k])

        return sample



def get_multi_dataset(img_path: str = "/workspace/twohand_full",
                      label_path: str = "/workspace/twohand_crop",
                      split: str = 'train',
                      capture_list: Optional[List[int]] = None,
                      need_image: bool = True,
                      need_camera: bool = True,
                      hand_type: str ='interacting',
                      T=15,
                      fps='5fps') -> InterHandSingleFrameDataset:
    assert split in ['train', 'test', 'val']
    dataset = InterHandSingleFrameDataset(
        camera_path=os.path.join(label_path, 'annotations', split, 'InterHand2.6M_' + split + '_camera.json'),
        label_path=os.path.join(label_path, 'combine_anno', split),
        image_path= os.path.join(img_path,'crop_images',split),
        capture_list=capture_list,
        need_image=need_image,
        need_camera=need_camera,
        split=split,
        fps=fps,
        hand_type=hand_type,
        T_len=T)
    return dataset


def bbox_2_square(bbox, ratio=0.75):
    x_min = bbox[..., 0]
    y_min = bbox[..., 1]
    x_max = bbox[..., 0] + bbox[..., 2]
    y_max = bbox[..., 1] + bbox[..., 3]

    mid = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2], dim=-1)  # ... x 2
    L = torch.max(torch.cat([(x_max - x_min), (y_max - y_min)], dim=-1), dim=-1).values[...,None] # bs x 1

    L = L / ratio

    bbox = torch.cat([mid[...,0] - L / 2, mid[...,1] - L / 2, L, L], dim=-1).int()

    return bbox

def crop_img(img, bbox):
    if isinstance(bbox,torch.Tensor):
        bbox = bbox.cpu().numpy()

    left = bbox[0]
    up = bbox[1]
    right = bbox[0] + bbox[2]
    bottom = bbox[1] + bbox[3]
    crop_img = img[max(up, 0):min(bottom, img.shape[0]), max(left, 0):min(right, img.shape[1])]
    crop_img = cv.copyMakeBorder(crop_img,
                                 max(-up, 0), max(bottom - img.shape[0], 0),
                                 max(-left, 0), max(right - img.shape[1], 0),
                                 cv.BORDER_CONSTANT, None, (0,0,0))
    return crop_img

def sub_img(img,img_bbox,sub_bbox): #img: 1 x 3 x W x H
    img_bbox = img_bbox.clone().int()
    sub_bbox = sub_bbox.clone().int()
    out_img = torch.zeros((1,3,int(sub_bbox[2]),int(sub_bbox[3])))
    sup_start = [max(0,int(sub_bbox[0]-img_bbox[0])),max(0,int(sub_bbox[1]-img_bbox[1]))]
    sup_end = [min(int(img_bbox[2]),int(sub_bbox[0]+sub_bbox[2]-img_bbox[0])),min(int(img_bbox[3]),int(sub_bbox[1]+sub_bbox[3]-img_bbox[1]))]

    out_start = [max(0,int(img_bbox[0]-sub_bbox[0])),max(0,int(img_bbox[1]-sub_bbox[1]))]
    out_end = [min(int(sub_bbox[2]),int(img_bbox[0]+img_bbox[2]-sub_bbox[0])),min(int(sub_bbox[3]),int(img_bbox[1]+img_bbox[3]-sub_bbox[1]))]
    out_img[:,:,out_start[1]:out_end[1],out_start[0]:out_end[0]] = img[:,:,sup_start[1]:sup_end[1],sup_start[0]:sup_end[0]]
    return out_img

def distance_map(bboxes):   # bs x 2 x 4
    bs = bboxes.shape[0]
    bbox_1 = bboxes[:,0,:]
    bbox_2 = bboxes[:,1,:]

    def generate_matrix(size, bs):
        x_values = torch.arange(0, size).unsqueeze(0).expand(size, -1) / size
        y_values = torch.arange(0, size).unsqueeze(1).expand(-1, size) / size
        return torch.stack([x_values, y_values], dim=2).unsqueeze(0).repeat(bs,1,1,1)
    map_1 = generate_matrix(16,bs)
    map_2 = generate_matrix(16,bs)
    map_1[..., 0] = map_1[..., 0] * bbox_1[..., 2].unsqueeze(1).unsqueeze(1) + bbox_1[..., 0].unsqueeze(1).unsqueeze(1)
    map_1[..., 1] = map_1[..., 1] * bbox_1[..., 3].unsqueeze(1).unsqueeze(1) + bbox_1[..., 1].unsqueeze(1).unsqueeze(1)
    map_2[..., 0] = map_2[..., 0] * bbox_2[..., 2].unsqueeze(1).unsqueeze(1) + bbox_2[..., 0].unsqueeze(1).unsqueeze(1)
    map_2[..., 1] = map_2[..., 1] * bbox_2[..., 3].unsqueeze(1).unsqueeze(1) + bbox_2[..., 1].unsqueeze(1).unsqueeze(1)

    map_1to2 = (map_1 - map_2) / bbox_1[..., 2].unsqueeze(1).unsqueeze(1)
    map_2to1 = (map_2 - map_1) / bbox_2[..., 2].unsqueeze(1).unsqueeze(1)

    return map_1to2, map_2to1



def Rmat_to_axis(Rmat, eps=1e-12):
    # Rmat: ... x 3 x 3
    # axis: ... x 3
    cos = -0.5 + 0.5 * (Rmat[..., 0, 0] + Rmat[..., 1, 1] + Rmat[..., 2, 2]).unsqueeze(-1)  # ... x 1
    angle = torch.acos(cos.clamp(min=-1+eps, max=1-eps)).clamp(min=eps, max=torch.pi-eps)  # ... x 1
    sin_x_L = 0.5 * (Rmat - Rmat.transpose(-1, -2))  # ... x 3 x 3
    sin_x_axes = sin_x_L[..., [2, 0, 1], [1, 2, 0]]  # ... x 3

    axes = torch.nn.functional.normalize(sin_x_axes)

    # when angle is close to PI, sin is close to zero and the normalize of sin_x_axes is not stable
    idx = angle[..., 0] > (torch.pi - 3e-3)
    if idx.any():
        xxt = (0.5 * (Rmat[idx] + Rmat[idx].transpose(-1, -2)) - cos[idx].unsqueeze(-1) * torch.eye(3).to(Rmat)) / (1 - cos[idx].unsqueeze(-1)).clamp_min(eps)
        axes_abs = torch.sqrt(torch.stack([xxt[..., 0, 0], xxt[..., 1, 1], xxt[..., 2, 2]], dim=-1).clamp_min(eps))
        axes_sign = axes[idx, :].sign()
        if (axes_sign == 0).any():
            xy_sign = xxt[..., 0, 1].sign()
            yz_sign = xxt[..., 1, 2].sign()
            zx_sign = xxt[..., 0, 2].sign()
            x_sign = (xy_sign > 0) | (zx_sign > 0)
            y_sign = (yz_sign > 0) | (xy_sign > 0)
            z_sign = (yz_sign > 0) | (zx_sign > 0)
            axes_sign = torch.stack([x_sign, y_sign, z_sign], dim=-1).float() * 2 - 1
        axes[idx] = axes_abs * axes_sign

    return axes * angle

def merge_bbox(bbox_list):
    bbox = np.stack(bbox_list, axis=0)  # N x ... x 4
    x_min = np.min(bbox[..., 0], axis=0)
    y_min = np.min(bbox[..., 1], axis=0)
    x_max = np.max(bbox[..., 0] + bbox[..., 2], axis=0)
    y_max = np.max(bbox[..., 1] + bbox[..., 3], axis=0)
    bbox = np.stack([x_min, y_min, x_max - x_min, y_max - y_min], axis=-1).astype(np.int32)
    return bbox

