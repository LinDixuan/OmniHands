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


class Relighted_Multiview(torch.utils.data.Dataset):
    def __init__(self,
                 data_path='/workspace/twohand_full/relighted_video',
                 task='train',
                 T=5):
        self.mano = MANO()
        self.data_path = data_path
        self.img_size = 256
        self.seq_len = T
        self.scale_factor = 1.75
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)
        self.split = 'train'
        assert task in ['train', 'val']
        if task == 'train':
            sequence_path = os.path.join(data_path, 'Sequences_train.json')
            bbox_path = os.path.join(data_path, 'bbox_train.json')
        else:
            sequence_path = os.path.join(data_path, 'Sequences_val.json')
            bbox_path = os.path.join(data_path, 'bbox_val.json')
        with open(sequence_path, 'r') as f:
            self.sequence_data = json.load(f)
        with open(bbox_path, 'r') as f:
            self.bbox_data = json.load(f)

        self.seq_size = []
        for i in range(len(self.sequence_data)):
            self.seq_size.append(len(self.sequence_data[f"{i}"]["img_names"]))
        self.seq_size = np.array(self.seq_size)
        self.seq_cumsum = np.cumsum(self.seq_size)
        self.data_size = self.seq_size.sum()

        self.camera_data = {}  # dir -> cid
        self.mano_data = {}

        dir_list = [dir for dir in os.listdir(data_path) if 'm--' in dir]

        for dir in dir_list:
            cam_path = os.path.join(self.data_path, dir, 'Mugsy_cameras', 'cam_params.json')
            with open(cam_path, 'r') as f:
                cam_data = json.load(f)
            self.camera_data[dir] = cam_data
            cam_views = os.listdir(
                os.path.join(cam_path[:cam_path.find("/cam_params")], 'envmap_per_segment', 'processed'))
            with open(os.path.join(cam_path[:cam_path.find('cam_params')], f"param_{cam_views[0]}.json"), 'r') as f:
                mano_data = json.load(f)
            self.mano_data[dir] = mano_data


    def __len__(self):
        return self.data_size

    def get_single(self, fname, mano_datas, cam_param, cam_id, bbox, image_path):
        img_name = fname

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

        output = {'left': {}, 'right': {}, 'image': img}
        joint_img, joint_cam, mesh_img, mesh_cam, pose, shape = get_mano_data(mano_param=mano_data['left'],
                                                                              cam_param=cam_data, do_flip=False,
                                                                              img_shape=(256, 256), mano=self.mano)

        output['left']['mano_param'] = np.concatenate([pose.reshape(48), shape.reshape(10)], axis=0)
        output['left']['joints2d'] = joint_img
        output['left']['joints3d'] = joint_cam
        output['left']['verts2d'] = mesh_img
        output['left']['verts3d'] = mesh_cam
        joint_img, joint_cam, mesh_img, mesh_cam, pose, shape = get_mano_data(mano_param=mano_data['right'],
                                                                              cam_param=cam_data, do_flip=False,
                                                                              img_shape=(256, 256), mano=self.mano)

        output['right']['mano_param'] = np.concatenate([pose.reshape(48), shape.reshape(10)], axis=0)
        output['right']['joints2d'] = joint_img
        output['right']['joints3d'] = joint_cam
        output['right']['verts2d'] = mesh_img
        output['right']['verts3d'] = mesh_cam

        return output

    def __getitem__(self, index):
        seq_id = np.searchsorted(self.seq_cumsum, index, side='right')
        seq_offset = 0 if seq_id == 0 else self.seq_cumsum[seq_id - 1]
        idx = index - seq_offset

        seq_data = self.sequence_data[f"{seq_id}"]
        fname = seq_data["img_names"][idx]
        cam_path = seq_data["cam_path"]

        image_path = os.path.join(cam_path[:cam_path.find("/cam_params")], 'envmap_per_segment', 'processed')
        image_path = os.path.join(self.data_path, image_path[image_path.find('m--'):])
        cam_all = os.listdir(image_path)
        cam_list = cam_all * 16
        random.shuffle(cam_list)

        dir_name = seq_data["img_path"]
        dir_name = dir_name[dir_name.find('m--'):dir_name.find('/Mugsy')]
        mano_datas = self.mano_data[dir_name]
        cam_datas = self.camera_data[dir_name]
        sample = collections.defaultdict(list)

        for cam_id in cam_list:
            cam_data = cam_datas[cam_id]
            cam_data = {k: np.array(v, dtype=np.float32) for k, v in cam_data.items()}
            cam_data['t'] /= 1000

            bbox = np.array(self.bbox_data[f"{seq_id}"][f"{cam_id}"], dtype=np.int32)

            output = self.get_single(fname, mano_datas, cam_data, cam_id, bbox, image_path)

            img = output["image"]
            joints2d_left = output["left"]["joints2d"]
            mano_param_left = output["left"]["mano_param"]
            joints2d_right = output["right"]["joints2d"]
            mano_param_right = output["right"]["mano_param"]
            verts2d_right = output['right']['verts2d']
            verts2d_left = output['left']['verts2d']
            joints3d_left = output["left"]["joints3d"]
            joints3d_right = output["right"]["joints3d"]
            verts3d_left = output["left"]["verts3d"]
            verts3d_right = output["right"]["verts3d"]

            crop_hand_right = get_bbox_joints(verts2d_right, bbox_factor=self.scale_factor)
            crop_hand_left = get_bbox_joints(verts2d_left, bbox_factor=self.scale_factor)
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

            joints2d_right = (joints2d_right - right_corner[None, :2]) / scale_right - 0.5
            joints2d_left = (joints2d_left - left_corner[None, :2]) / scale_left - 0.5

            img_right = img.crop(right_corner.tolist())
            img_right = img_right.resize((256, 256))
            img_left = img.crop(left_corner.tolist())
            img_left = img_left.resize((256, 256))
            img_full = img.resize((256, 256))
            bbox_hand_right = bbox_right
            bbox_hand_left = bbox_left
            bbox_full = np.zeros_like(bbox_hand_right)
            bbox_full[2], bbox_full[3] = 256, 256

            img_right = torchvision.transforms.functional.to_tensor(img_right).permute(1, 2, 0).flip(2) * 255
            img_left = torchvision.transforms.functional.to_tensor(img_left).permute(1, 2, 0).flip(2) * 255
            img_full = torchvision.transforms.functional.to_tensor(img_full).permute(1, 2, 0).flip(2) * 255
            for n_c in range(3):
                img_right[:, :, n_c] = (img_right[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
                img_left[:, :, n_c] = (img_left[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]

            # -------flip-------
            pose_left = mano_param_left[:48].reshape(16, 3)
            pose_left[:, 1:3] = -pose_left[:, 1:3]
            mano_param_left[:48] = pose_left.reshape(48)
            img_left = img_left.flip(1)
            joints2d_left[:, 0] = - joints2d_left[:, 0]
            # ------------------
            sample["img_left"].append(img_left)
            sample["bbox_left"].append(bbox_hand_left)
            sample["joints2d_left"].append(joints2d_left)
            sample["joints3d_world_left"].append(joints3d_left)
            sample["verts3d_world_left"].append(verts3d_left)
            sample["mano_param_left"].append(mano_param_left)
            sample["mano_pose_left"].append(mano_param_left[:48])
            sample["mano_shape_left"].append(mano_param_left[48:])

            sample["img_right"].append(img_right)
            sample["bbox_right"].append(bbox_hand_right)
            sample["joints2d_right"].append(joints2d_right)
            sample["joints3d_world_right"].append(joints3d_right)
            sample["verts3d_world_right"].append(verts3d_right)
            sample["mano_param_right"].append(mano_param_right)
            sample["mano_pose_right"].append(mano_param_right[:48])
            sample["mano_shape_right"].append(mano_param_right[48:])

            sample["img_full"].append(img_full)
            sample["bbox_full"].append(bbox_full)
            if len(sample['img_right']) >= self.seq_len:
                break

        for k in sample.keys():
            if isinstance(sample[k][0], np.ndarray):
                sample[k] = np.stack(sample[k], axis=0)
                sample[k] = torch.FloatTensor(sample[k])
            elif isinstance(sample[k][0], torch.Tensor):
                sample[k] = torch.stack(sample[k], dim=0)
            else:
                sample[k] = torch.FloatTensor(sample[k])

        return sample


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
            file.write('{} {} {}\n'.format(vertex_data[i, 0], vertex_data[i, 1], vertex_data[i, 2]))

        for i in range(num_faces):
            file.write('3 {} {} {}\n'.format(face_data[i, 0], face_data[i, 1], face_data[i, 2]))


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
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
    L = torch.max(torch.cat([(x_max - x_min), (y_max - y_min)], dim=-1), dim=-1).values[..., None]  # bs x 1

    L = L / ratio

    bbox = torch.cat([mid[..., 0] - L / 2, mid[..., 1] - L / 2, L, L], dim=-1).int()

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
    if isinstance(bbox, torch.Tensor):
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
    out_img = torch.zeros(bs, bbox[2], bbox[3], 3).to(img)

    L1 = min(H, bottom) - max(0, up)
    L2 = min(W, right) - max(0, left)

    out_img[:, max(0, -up):max(0, -up) + L1, max(0, -left):max(0, -left) + L2, :] \
        = img[:, max(0, up):max(0, up) + L1, max(0, left):max(0, left) + L2, :]
    out_img = torch.nn.functional.interpolate(out_img.permute(0, 3, 1, 2), size=img_size)
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
    def __init__(self, dataset, batch_size):
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


def process_crop_twohand(data_path='/data/relighted', task='train'):
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

        with open(param_path, 'r') as f:
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
                right_joint, _, right_mesh, _, _, _ = get_mano_data(param_right, cam_data, False, image_shape, mano)

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
                                           'right_path': save_path_right, 'left_path': save_path_left, 'dir': dir_id,
                                           'cam_view': cam_view, 'img_name': img_name}
                data_counter += 1
    with open(os.path.join(save_path, f"crop_data.json"), 'w') as f:
        json.dump(crop_info, f)
    with open(os.path.join(save_path, f"dir_dict.json"), 'w') as f:
        json.dump(dir_dict, f)


if __name__ == '__main__':
    data_path = '/data/relighted'
    dataset = Relighted_Hand(data_path)
