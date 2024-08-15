import torch
import torchvision
import os
import sys
import pickle
import json
import numpy as np
import cv2 as cv
import random
import tqdm
from .relighted_mano import get_mano_data, MANO, get_mano_data_old
import time
from .utils import flip_pose, expand_to_aspect_ratio
from PIL import Image, ImageFilter, ImageOps
from PIL import ImageFile
import collections
from .ho3d_util import *
from .dataset_util import *


class Relighted_Hand_Video_V3(torch.utils.data.Dataset):
    def __init__(self,
                 data_path='/workspace/twohand_full/relighted_video',
                 task='train',
                 seg_len=15,
                 gap=3):
        self.mano = MANO()
        self.data_path = data_path
        self.img_size = 256
        self.seg_len = seg_len
        self.gap = gap
        self.scale_factor = 2.0
        IMAGE_MEAN = [0.485,0.456,0.406]
        IMAGE_STD  = [0.229,0.224,0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)
        assert task in ['train','val']
        if task == 'train':
            sequence_path = os.path.join(data_path,'Sequences_train.json')
            bbox_path = os.path.join(data_path,'bbox_train.json')
        else:
            sequence_path = os.path.join(data_path,'Sequences_val.json')
            bbox_path = os.path.join(data_path, 'bbox_val.json')
        with open(sequence_path,'r') as f:
            self.sequence_data = json.load(f)
        with open(bbox_path,'r') as f:
            self.bbox_data = json.load(f)

        seq_len = []
        for i in range(len(self.sequence_data)):
            seq_len.append(len(self.sequence_data[f"{i}"]["img_names"]))
        prob = np.array(seq_len,dtype=np.float32)
        prob = prob / np.sum(prob)
        self.prob = np.cumsum(prob)
        self.prob[-1] = 1.

        self.camera_data = {}   # dir -> cid
        self.mano_data = {}

        dir_list = [dir for dir in os.listdir(data_path) if 'm--' in dir]
        #if task == 'train':
        #    dir_list = dir_list[:-1]
        #else:
        #    dir_list = dir_list[-1:]

        for dir in dir_list:
            cam_path = os.path.join(self.data_path, dir, 'Mugsy_cameras', 'cam_params.json')
            with open(cam_path,'r') as f:
                cam_data = json.load(f)
            self.camera_data[dir] = cam_data
            cam_views = os.listdir(os.path.join(cam_path[:cam_path.find("/cam_params")], 'envmap_per_segment', 'processed'))
            with open(os.path.join(cam_path[:cam_path.find('cam_params')],f"param_{cam_views[0]}.json"),'r') as f:
                mano_data = json.load(f)
            self.mano_data[dir] = mano_data


    def get_sample(self):
        dice = random.random()
        seq_id = np.searchsorted(self.prob, dice, side='right')

        seq_data = self.sequence_data[f"{seq_id}"]

        cam_path = seq_data["cam_path"]

        image_path = os.path.join(cam_path[:cam_path.find("/cam_params")], 'envmap_per_segment', 'processed')

        image_path = os.path.join(self.data_path, image_path[image_path.find('m--'):])

        cam_list = os.listdir(image_path)

        cam_id = random.choice(range(len(cam_list)))

        select_seg = self.select_seg(seq_data, cam_id, seq_id, self.gap)

        return select_seg


    def __len__(self):
        return 450_000

    def data_aug(
            self,
            img,
            mano_param_right,
            mano_param_left,
            joints2d_right,
            joints2d_left,
            joints3d_right,
            joints3d_left,
            verts3d_right,
            verts3d_left,
            center_rand,
            scale_rand,
            rot_rand,
            blur_rand,
            color_jitter_funcs,
            mask_rand
    ):
        center_rand = np.random.uniform(low=-1, high=1, size=2)
        scale_rand = np.random.randn()
        outside_mask_right = ((joints2d_right < 0) + (joints2d_right > self.img_size)).any(axis=1)
        outside_mask_left = ((joints2d_left < 0) + (joints2d_left > self.img_size)).any(axis=1)

        padded_size = 384
        padding = padded_size - 256
        joints2d_right += padding
        joints2d_left += padding
        img = ImageOps.expand(img, border=padding, fill='black')
        crop_hand_right = get_bbox_joints(joints2d_right, bbox_factor=2.0)
        crop_hand_left = get_bbox_joints(joints2d_left, bbox_factor=2.0)

        center, scale = fuse_bbox(crop_hand_right, crop_hand_left, img.size)

        rot = rot_rand
        affinetrans, rot_mat = get_affine_transform(
            center, scale, [padded_size, padded_size], rot=rot
        )

        # Change mano from openGL coordinates to normal coordinates
        mano_param_right[:3] = rotation_angle(mano_param_right[:3], rot_mat)
        mano_param_left[:3] = rotation_angle(mano_param_left[:3], rot_mat)

        joints3d_right = rot_mat.dot(joints3d_right.transpose()).transpose()
        joints3d_left = rot_mat.dot(joints3d_left.transpose()).transpose()
        verts3d_right = rot_mat.dot(verts3d_right.transpose()).transpose()
        verts3d_left = rot_mat.dot(verts3d_left.transpose()).transpose()

        joints2d_right = transform_coords(
            joints2d_right, affinetrans
        )  # hand landmark trans
        joints2d_left = transform_coords(
            joints2d_left, affinetrans
        )  # hand landmark trans
        joints2d_glb_right = joints2d_right.copy()
        joints2d_glb_left = joints2d_left.copy()
        # ------------crop hand--------------
        crop_hand_right = get_bbox_joints(joints2d_right, bbox_factor=self.scale_factor)
        crop_hand_left = get_bbox_joints(joints2d_left, bbox_factor=self.scale_factor)
        center_right = (crop_hand_right[:2] + crop_hand_right[2:]) / 2
        center_left = (crop_hand_left[:2] + crop_hand_left[2:]) / 2
        scale_right = crop_hand_right[2:] - crop_hand_right[:2]
        scale_right = expand_to_aspect_ratio(scale_right, [192, 256])
        scale_left = crop_hand_left[2:] - crop_hand_left[:2]
        scale_left = expand_to_aspect_ratio(scale_left, [192, 256])
        crop_hand_right[2:] = center_right + scale_right / 2
        crop_hand_left[2:] = center_left + scale_left / 2
        crop_hand_right[:2] = center_right - scale_right / 2
        crop_hand_left[:2] = center_left - scale_left / 2

        center_right, scale_right = fuse_bbox(crop_hand_right, crop_hand_right, img.size)
        center_left, scale_left = fuse_bbox(crop_hand_left, crop_hand_left, img.size)

        #scale_left = scale_right = max(scale_left, scale_right)

        # Scale jittering
        scale_jittering = 0.3 * scale_rand + 1
        scale_jittering = np.clip(
            scale_jittering, 1 - 0.3, 1 + 0.3
        )
        scale_right = scale_right * scale_jittering
        scale_left = scale_left * scale_jittering

        # Randomly jitter center
        center_offsets = 0.1 * scale_right * center_rand
        center_right = center_right + center_offsets
        center_left = center_left + center_offsets

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

        bbox_right = np.array([*(center_right - scale_right / 2), scale_right, scale_right], dtype=np.int32)
        bbox_left = np.array([*(center_left - scale_left / 2), scale_left, scale_left], dtype=np.int32)
        right_corner = bbox_right.copy()
        right_corner[2:] = right_corner[2:] + right_corner[:2]
        left_corner = bbox_left.copy()
        left_corner[2:] = left_corner[2:] + left_corner[:2]

        joints2d_right = joints2d_right / 256 - 0.5
        joints2d_left = joints2d_left / 256 - 0.5
        joints2d_glb_right = joints2d_glb_right / 384 - 0.5
        joints2d_glb_left = joints2d_glb_left / 384 - 0.5
        joints2d_right[outside_mask_right] = -1
        joints2d_left[outside_mask_left] = -1
        joints2d_glb_right[outside_mask_right] = -1
        joints2d_glb_left[outside_mask_left] = -1
        # ------------------------------------

        # Transform and crop
        img = transform_img(img, affinetrans, [padded_size, padded_size])

        img_right = img.crop(right_corner.tolist())
        img_right = img_right.resize((256, 256))
        img_left = img.crop(left_corner.tolist())
        img_left = img_left.resize((256, 256))

        # Img blurring and color jitter
        blur_radius = blur_rand * 1.0
        img_right = img_right.filter(ImageFilter.GaussianBlur(blur_radius))
        img_left = img_left.filter(ImageFilter.GaussianBlur(blur_radius))

        for func in color_jitter_funcs:
            img_right = func(img_right)
            img_left = func(img_left)


        return (img, img_right, img_left, mano_param_right, mano_param_left,
                joints2d_right, joints2d_left,
                joints2d_glb_right,
                joints2d_glb_left,
                joints3d_right, joints3d_left,
                verts3d_right, verts3d_left, bbox_right, bbox_left)

    def get_single(self, index, seq_data, mano_datas, cam_param, cam_id, bbox, image_path):
        img_id = index % 10000
        img_name = seq_data["img_names"][img_id]

        mano_data = mano_datas[f"{img_name}"]
        mano_data['left']['hand_type'] = 'left'
        mano_data['right']['hand_type'] = 'right'
        img_path = os.path.join(image_path, cam_id, mano_data["image"].split('/')[-1])
        img = Image.open(img_path).convert("RGB")

        cam_data = {}
        for k in cam_param.keys():
            cam_data[k] = cam_param[k].copy()
        cam_data['princpt'][0] -= bbox[0]
        cam_data['princpt'][1] -= bbox[1]
        cam_data['focal'] = cam_data['focal'] * self.img_size / bbox[2]
        cam_data['princpt'] = cam_data['princpt'] * self.img_size / bbox[2]

        output = {'left':{},'right':{},'image':img}
        joint_img, joint_cam, mesh_img, mesh_cam, pose, shape = get_mano_data(mano_param=mano_data['left'],
                                                                cam_param=cam_data,do_flip=False,img_shape=(256,256),mano=self.mano)
        #pose = batch_rodrigues((torch.from_numpy(pose) + self.mano.layer['left'].pose_mean).view(-1, 3)).view(-1, 3, 3)

        output['left']['mano_param'] = np.concatenate([pose.reshape(48), shape.reshape(10)], axis=0)
        output['left']['joints2d'] = joint_img
        output['left']['joints3d'] = joint_cam
        output['left']['verts3d'] = mesh_cam
        joint_img, joint_cam, mesh_img, mesh_cam, pose, shape = get_mano_data(mano_param=mano_data['right'],
                                                                              cam_param=cam_data, do_flip=False,
                                                                              img_shape=(256, 256), mano=self.mano)

        #pose = batch_rodrigues((torch.from_numpy(pose)+self.mano.layer['right'].pose_mean).view(-1,3)).view(-1,3,3)

        output['right']['mano_param'] = np.concatenate([pose.reshape(48), shape.reshape(10)], axis=0)
        output['right']['joints2d'] = joint_img
        output['right']['joints3d'] = joint_cam
        output['right']['verts3d'] = mesh_cam

        return output

    def __getitem__(self, index):
        seq_ids = self.get_sample()
        seq_id = seq_ids[0]
        seq_id = seq_id // 100000
        cam_id = (seq_id // 10000) % 10

        seq_data = self.sequence_data[f"{seq_id}"]

        dir_name = seq_data["img_path"]
        dir_name = dir_name[dir_name.find('m--'):dir_name.find('/Mugsy')]

        cam_path = seq_data["cam_path"]
        cam_path = cam_path[cam_path.find('m--'):cam_path.find("/cam_params")]
        cam_datas = self.camera_data[dir_name]

        image_path = os.path.join(self.data_path, cam_path, 'envmap_per_segment', 'processed')
        cam_list = os.listdir(image_path)

        cam_id = cam_list[cam_id]
        mano_datas = self.mano_data[dir_name]
        cam_data = cam_datas[cam_id]
        cam_data = {k: np.array(v, dtype=np.float32) for k, v in cam_data.items()}
        cam_data['t'] /= 1000

        bbox = np.array(self.bbox_data[f"{seq_id}"][f"{cam_id}"], dtype=np.int32)

        sample = collections.defaultdict(list)
        flip = random.random() > 0.5

        center_rand = np.random.uniform(low=-1, high=1, size=2)
        scale_rand = np.random.randn()
        rot_rand = np.random.uniform(low=-np.pi, high=np.pi)
        blur_rand = random.random()
        motion_rand = random.random()
        color_jitter_funcs = get_color_jitter_funcs(
            brightness=random.random() + 0.5,
        )
        mask_rand = random.random()

        for idx in seq_ids:
            output = self.get_single(idx, seq_data, mano_datas, cam_data, cam_id, bbox, image_path)

            img = output["image"]
            joints2d_left = output["left"]["joints2d"]
            mano_param_left = output["left"]["mano_param"]
            joints2d_right = output["right"]["joints2d"]
            mano_param_right = output["right"]["mano_param"]
            joints3d_left = output["left"]["joints3d"]
            joints3d_right = output["right"]["joints3d"]
            verts3d_left = output["left"]["verts3d"]
            verts3d_right = output["right"]["verts3d"]

            (
                img_full,
                img_right,
                img_left,
                mano_param_right,
                mano_param_left,
                joints2d_right,
                joints2d_left,
                joints2d_glb_right,
                joints2d_glb_left,
                joints3d_right,
                joints3d_left,
                verts3d_right,
                verts3d_left,
                bbox_hand_right,
                bbox_hand_left,
            ) = self.data_aug(
                img,
                mano_param_right,
                mano_param_left,
                joints2d_right,
                joints2d_left,
                joints3d_right,
                joints3d_left,
                verts3d_right,
                verts3d_left,
                center_rand,
                scale_rand,
                rot_rand,
                blur_rand,
                color_jitter_funcs,
                mask_rand
            )

            img_right = torchvision.transforms.functional.to_tensor(img_right).permute(1, 2, 0).flip(2) * 255
            img_left = torchvision.transforms.functional.to_tensor(img_left).permute(1, 2, 0).flip(2) * 255
            img_full = torchvision.transforms.functional.to_tensor(img_full).permute(1, 2, 0).flip(2) * 255
            for n_c in range(3):
                img_right[:, :, n_c] = (img_right[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
                img_left[:, :, n_c] = (img_left[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]

            bbox_full = np.zeros_like(bbox_hand_right)
            bbox_full[2], bbox_full[3] = 384, 384

            # -------flip-------
            pose_left = mano_param_left[:48].reshape(16, 3)
            pose_left[:, 1:3] = -pose_left[:, 1:3]
            mano_param_left[:48] = pose_left.reshape(48)
            img_left = img_left.flip(1)
            joints2d_left[:, 0] = - joints2d_left[:, 0]
            joints2d_glb_left[:, 0] = -joints2d_glb_left[:, 0]
            # ------------------
            sample["img_left"].append(img_left)
            sample["bbox_left"].append(bbox_hand_left)
            sample["joints2d_left"].append(joints2d_left)
            #sample["joints2d_glb_left"].append(joints2d_glb_left)
            sample["joints3d_world_left"].append(joints3d_left)
            sample["verts3d_world_left"].append(verts3d_left)
            sample["mano_param_left"].append(mano_param_left)
            sample["mano_pose_left"].append(mano_param_left[:48])
            sample["mano_shape_left"].append(mano_param_left[48:])

            sample["img_right"].append(img_right)
            sample["bbox_right"].append(bbox_hand_right)
            sample["joints2d_right"].append(joints2d_right)
            #sample["joints2d_glb_right"].append(joints2d_glb_right)
            sample["joints3d_world_right"].append(joints3d_right)
            sample["verts3d_world_right"].append(verts3d_right)
            sample["mano_param_right"].append(mano_param_right)
            sample["mano_pose_right"].append(mano_param_right[:48])
            sample["mano_shape_right"].append(mano_param_right[48:])

            sample["img_full"].append(img_full)
            sample["bbox_full"].append(bbox_full)
            sample["inter"].append(1)

        for k in sample.keys():
            if isinstance(sample[k][0], np.ndarray):
                sample[k] = np.stack(sample[k],axis=0)
                sample[k] = torch.FloatTensor(sample[k])
            elif isinstance(sample[k][0],torch.Tensor):
                sample[k] = torch.stack(sample[k])
            else:
                sample[k] = torch.FloatTensor(sample[k])

        sample['dataset'] = torch.ones(self.seg_len) * 2
                
        """for fid in range(sample['img_right'].shape[0]):
            if random.random() < 0.15:
                sample['img_right'][fid] = torch.zeros_like(sample['img_right'][fid])
                sample['img_left'][fid] = torch.zeros_like(sample['img_left'][fid])"""

        """if random.random() > 0.75:
            img_right = sample["img_right"]
            img_left = sample["img_left"]
            radius = min(3, self.seg_len // 2)
            motion_w = get_fw_weight(radius, motion_rand * 0.8)
            motion_w = torch.from_numpy(motion_w)[:, None, None, None]
            for fid in range(radius, self.seg_len):
                img_right[fid] = (img_right[fid - radius: fid + 1] * motion_w).sum(dim=0)
                img_left[fid] = (img_left[fid - radius: fid + 1] * motion_w).sum(dim=0)
            sample["img_right"] = img_right
            sample["img_left"] = img_left"""

        return sample

    def select_seg(self, seq_data, cam_id, seq_id, gap):
        select_ids = []
        dice = random.random()
        pos = int(dice * (len(seq_data["img_names"]) - 1))
        dice = random.random()
        inverse = dice > 0.7
        for i in range(self.seg_len):
            select_ids.append(int(pos) + int(cam_id) * 10000 + int(seq_id) * 100000)
            if inverse:
                if pos - gap < 0:
                    inverse = False
                    pos = pos + gap
                else:
                    pos = pos - gap
            else:
                if pos + gap >= len(seq_data["img_names"]):
                    inverse = True
                    pos = pos - gap
                else:
                    pos = pos + gap
        return select_ids

def save_mesh_to_ply(vertex_data, face_data, file_path):
    num_vertices = vertex_data.shape[0]
    num_faces = face_data.shape[0]

    with open(file_path, 'w') as file:
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {}\n'.format(num_vertices))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('element face {}\n'.format(num_faces))
        file.write('property list uchar int vertex_indices\n')
        file.write('end_header\n')

        for i in range(num_vertices):
            file.write('{} {} {}\n'.format(vertex_data[i,0],vertex_data[i,1],vertex_data[i,2]))

        for i in range(num_faces):
            file.write('3 {} {} {}\n'.format(face_data[i,0],face_data[i,1],face_data[i,2]))


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


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

def bbox_2_square_torch(bbox, ratio=0.75):
    x_min = bbox[..., 0]
    y_min = bbox[..., 1]
    x_max = bbox[..., 0] + bbox[..., 2]
    y_max = bbox[..., 1] + bbox[..., 3]

    mid = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2], dim=-1)  # ... x 2
    L = torch.max(torch.cat([(x_max - x_min), (y_max - y_min)], dim=-1), dim=-1).values[...,None] # bs x 1

    L = L / ratio

    bbox = torch.cat([mid[...,0] - L / 2, mid[...,1] - L / 2, L, L], dim=-1).int()

    return bbox


def merge_bbox(bbox_list):
    bbox = np.stack(bbox_list, axis=0)  # N x ... x 4
    x_min = np.min(bbox[..., 0], axis=0)
    y_min = np.min(bbox[..., 1], axis=0)
    x_max = np.max(bbox[..., 0] + bbox[..., 2], axis=0)
    y_max = np.max(bbox[..., 1] + bbox[..., 3], axis=0)
    bbox = np.stack([x_min, y_min, x_max - x_min, y_max - y_min], axis=-1).astype(np.int32)
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
                                 cv.BORDER_CONSTANT, None, (0, 0, 0))
    return crop_img

def crop_resize_img_batch(img, bbox, img_size):
    bs, H, W, C = img.shape
    left = bbox[0]
    up = bbox[1]
    right = bbox[0] + bbox[2]
    bottom = bbox[1] + bbox[3]
    out_img = torch.zeros(bs,bbox[2],bbox[3],3).to(img)

    L1 = min(H,bottom) - max(0,up)
    L2 = min(W,right) - max(0,left)

    out_img[:,max(0,-up):max(0, -up) + L1, max(0,-left):max(0,-left) + L2,:]\
        = img[:,max(0,up):max(0,up) + L1, max(0,left):max(0,left) + L2,:]
    out_img = torch.nn.functional.interpolate(out_img.permute(0,3,1,2),size=img_size)
    out_img = out_img / 255
    return out_img


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

class random_iter:
    def __init__(self,dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return 100_000_000

    def __iter__(self):
        return self

    def __next__(self):
        output = []
        for i in range(self.batch_size):
            output += self.dataset.get_sample()
        return output

class RandomSequenceSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        data_source = random_iter(dataset, batch_size)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)

def process_crop_twohand(data_path='/data/relighted',task='train'):
    mano = MANO()
    path_list = os.listdir(data_path)
    path_list = [path for path in path_list if 'm--' in path]

    data_counter = 0
    crop_info = {}
    dir_dict = {}
    save_path = '/data/twohand_crop/relighted_crop'
    for dir_id, dir in enumerate(path_list):

        if 'm--' not in dir:
            continue
        dir_dict[dir_id] = dir

        param_path = os.path.join('/data/relighted', dir, 'mano_fits', 'all.json')
        image_dir = os.path.join(data_path, dir, 'Mugsy_cameras', 'envmap_per_frame', 'images')
        cam_path = os.path.join(data_path, dir, 'Mugsy_cameras', 'cam_params.json')
        cam_list = os.listdir(image_dir)

        with open(cam_path, 'r') as f:
            cam_label = json.load(f)

        with open(param_path,'r') as f:
            params = json.load(f)

        for cam_view in cam_list:

            print(f"data count:{data_counter}")
            image_list = os.listdir(os.path.join(image_dir, cam_view))

            cam_data = cam_label[cam_view]
            cam_data = {k: np.array(v, dtype=np.float32) for k, v in cam_data.items()}
            cam_data['t'] /= 1000

            for p in tqdm.tqdm(sorted(image_list)):
                image_path = os.path.join(image_dir, cam_view, p)
                img_name = int(p.split('.')[0])

                param_left = params[f"{img_name}_left"]
                param_right = params[f"{img_name}_right"]

                param_left['hand_type'] = 'left'
                param_right['hand_type'] = 'right'

                image_shape = (1024, 667)  # H x W
                left_joint, _, left_mesh, _, _, _ = get_mano_data(param_left, cam_data, False, image_shape, mano)
                right_joint, _, right_mesh, _, _, _ = get_mano_data(param_right, cam_data, False, image_shape,mano)

                v2d_left, v2d_right = torch.from_numpy(left_mesh), torch.from_numpy(right_mesh)

                right_max = v2d_right.max(dim=0).values
                right_min = v2d_right.min(dim=0).values

                bbox_right = torch.FloatTensor(
                    [[right_min[0], right_min[1], right_max[0] - right_min[0], right_max[1] - right_min[1]]])
                bbox_right = bbox_2_square_torch(bbox_right, ratio=0.75)

                left_max = v2d_left.max(dim=0).values
                left_min = v2d_left.min(dim=0).values

                bbox_left = torch.FloatTensor(
                    [[left_min[0], left_min[1], left_max[0] - left_min[0], left_max[1] - left_min[1]]])
                bbox_left = bbox_2_square_torch(bbox_left, ratio=0.75)

                bboxes = torch.stack([bbox_right, bbox_left], dim=0).unsqueeze(0)

                img = cv.imread(image_path)

                crop_img_right = crop_img(img, bbox_right)
                crop_img_left = crop_img(img, bbox_left)[:, ::-1, :]

                dir_name = data_counter // 1000

                if not os.path.exists(os.path.join(save_path, f'{dir_name}')):
                    os.makedirs(os.path.join(save_path, f'{dir_name}'), exist_ok=True)
                save_path_right = os.path.join(save_path, f'{dir_name}', f'{data_counter}_right.jpg')
                save_path_left = os.path.join(save_path, f'{dir_name}', f'{data_counter}_left.jpg')

                cv.imwrite(save_path_right, crop_img_right)
                cv.imwrite(save_path_left, crop_img_left)
                crop_info[data_counter] = {'bbox_left': bbox_left.tolist(), 'bbox_right': bbox_right.tolist(),
                                'right_path': save_path_right, 'left_path': save_path_left,'dir':dir_id,'cam_view':cam_view,'img_name':img_name}
                data_counter += 1
    with open(os.path.join(save_path, f"crop_data.json"), 'w') as f:
        json.dump(crop_info, f)
    with open(os.path.join(save_path, f"dir_dict.json"), 'w') as f:
        json.dump(dir_dict, f)


if __name__ == '__main__':
    data_path = '/data/relighted'
    dataset = Relighted_Hand(data_path)
