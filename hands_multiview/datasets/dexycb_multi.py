from ..datasets import dataset_util
from ..datasets import ho3d_util
# import dataset_util
# import ho3d_util
import yaml
import torch
import os
import collections
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
import tqdm
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
from .utils import expand_to_aspect_ratio
from .manolayer_dexycb import ManoLayer

# from manolayer_dexycb import ManoLayer

_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]

_YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

_MANO_JOINTS = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]


class DexYCB_Temporal(data.Dataset):
    def __init__(
            self,
            dataset_root,
            setup="s0",
            train_label_root="./dexycb-process",
            mode="train",
            inp_res=256,
            T=15,
    ):
        # Dataset attributes
        self._data_dir = dataset_root
        self.train_label_root = train_label_root
        self._setup = setup
        self._split = mode
        self.inp_res = inp_res
        self.T = T
        self.dex_mano_right = ManoLayer(use_pca=True, flat_hand_mean=False, side="right", ncomps=45,
                                        mano_root="/workspace/hamer_twohand/_DATA/data/mano")
        self.dex_mano_left = ManoLayer(use_pca=True, flat_hand_mean=False, side="left", ncomps=45,
                                       mano_root="/workspace/hamer_twohand/_DATA/data/mano")
        self.coord_change_mat = np.array(
            [[1.0, 0.0, 0.0], [0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        self._model_dir = os.path.join(self._data_dir, "models")

        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._h = 480
        self._w = 640
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)
        if self._split == "train":
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 != 4]
        if self._split == "val":
            subject_ind = [0, 1]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        if self._split == "test":
            subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        self._subjects = [_SUBJECTS[i] for i in subject_ind]
        self._serials = [_SERIALS[i] for i in serial_ind]

        self._sequences = []
        self._mapping = []
        self.mul_ids = collections.defaultdict(list)
        offset = 0
        for n in self._subjects:
            seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
            seq = [os.path.join(n, s) for s in seq]
            assert len(seq) == 100
            seq = [seq[i] for i in sequence_ind]
            self._sequences += seq
            for i, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, "r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                c = np.arange(len(self._serials))
                f = np.arange(meta["num_frames"])
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = (offset + i) * np.ones_like(c)
                m = np.vstack((s, c, f)).T

                self._mapping.append(m)
            offset += len(seq)

        self._mapping = np.vstack(self._mapping)
        for ids in range(self._mapping.shape[0]):
            s, c, f = self._mapping[ids]
            self.mul_ids[f"S{s}F{f}"].append(ids)

        self.valid_idxs = []
        self.mano_params = []
        self.joints_uv = []
        self.mano_joints_uv = []
        self.K = []
        self.obj_p2ds = []

        self.mano_side_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_manoside.json")
        )
        mano_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_mano.json")
        )
        joints_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_joint.json")
        )
        mano_joint_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_Manojoint.json")
        )
        K_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_K.json")
        )
        obj_p2d_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_obj_c2d.json")
        )
        self.joint_list = joints_list
        assert len(mano_list) == len(self._mapping)
        for i in range(len(mano_list)):
            K = np.array(K_list[i], dtype=np.float32)
            self.K.append(K)
            joints_uv_np = np.array(joints_list[i], dtype=np.float32)
            joints_mano_np = np.array(mano_joint_list[i], dtype=np.float32)
            self.joints_uv.append(
                ho3d_util.projectPoints(joints_uv_np, K)
            )
            self.mano_joints_uv.append(
                ho3d_util.projectPoints(joints_mano_np - joints_mano_np[0:1] + joints_uv_np[0:1], K)
            )

            self.mano_params.append(np.array(mano_list[i], dtype=np.float32))
            self.obj_p2ds.append(np.array(obj_p2d_list[i], dtype=np.float32))
            if (self.mano_params[-1][:48] != 0).any():
                self.valid_idxs.append(i)
        self.sequence_list = self.get_sequence_list()

    def get_sequence_list(self):
        cache_filepath = os.path.join(
            self.train_label_root,
            f"{self._split}_temporal_windows_9_5.json",
        )

        return ho3d_util.json_load(cache_filepath)

    def data_crop(self, img, K, bbox_hand):
        crop_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.0
        )

        center, scale = dataset_util.fuse_bbox(crop_hand, crop_hand, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res]
        )

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)

        bbox_hand_corners = dataset_util.get_bbox_corners(bbox_hand)
        bbox_hand_corners = dataset_util.transform_coords(
            bbox_hand_corners, affinetrans
        )
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand_corners)
        bbox_hand = dataset_util.regularize_bbox(bbox_hand, img.size)
        return img, K, bbox_hand, affinetrans

    def __len__(self):
        return len(self.sequence_list) 

    def __getitem__(self, index):
        sample = collections.defaultdict(list)

        ss, _, ff = self._mapping[index]
        mul_ids = self.mul_ids[f'S{ss}F{ff}']
        mul_ids_ext = mul_ids * 2
        random.shuffle(mul_ids_ext)
        mul_ids = mul_ids + mul_ids_ext
        for id in mul_ids:
            img_idx = id
            s, c, f = self._mapping[img_idx]

            d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
            img_filename = os.path.join(d, self._color_format.format(f))
            if not os.path.exists(img_filename):
                continue
            img = Image.open(img_filename).convert("RGB")

            # camera information
            K = self.K[img_idx].copy()

            # hand information
            joints_uv = self.joints_uv[img_idx].copy()
            mano_joints_uv = self.mano_joints_uv[img_idx].copy()
            mano_param = self.mano_params[img_idx].copy()
            mano_side = self.mano_side_list[img_idx]

            # object information
            gray = None
            p2d = self.obj_p2ds[img_idx].copy()

            crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
            center = (crop_hand[:2] + crop_hand[2:]) / 2
            scale = crop_hand[2:] - crop_hand[:2]
            scale = expand_to_aspect_ratio(scale, [192, 256])
            crop_hand[2:] = center + scale / 2
            crop_hand[:2] = center - scale / 2
            crop_hand[2:] = crop_hand[:2] + scale
            bbox_hand = crop_hand
            img, K, bbox_hand, affinetrans = self.data_crop(img, K, bbox_hand)
            mano_joints_uv = dataset_util.transform_coords(
                mano_joints_uv, affinetrans
            )  # hand landmark trans
            mano_joints_uv = dataset_util.normalize_joints(mano_joints_uv, bbox_hand)

            sample["img"].append(functional.to_tensor(img))
            sample["bbox_hand"].append(bbox_hand)
            sample["mano_param"].append(mano_param)
            sample["mano_side"].append(mano_side)
            sample["joints2d"].append(mano_joints_uv)
            if len(sample["img"]) == self.T:
                break

        imgs = torch.stack(sample["img"], dim=0).flip(1).permute(0, 2, 3, 1) * 255
        imgs_cv = imgs.clone()
        for n_c in range(3):
            imgs[:, :, :, n_c] = (imgs[:, :, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
        bbox_hand = np.stack(sample["bbox_hand"], axis=0)
        mano_param = np.stack(sample["mano_param"], axis=0)
        mano_side = np.array(sample["mano_side"])

        joints2d = np.stack(sample["joints2d"], axis=0)
        bbox_hand_corner = bbox_hand.reshape(-1, 2, 2)
        joints2d = joints2d * (bbox_hand_corner[:, 1:2, :] - bbox_hand_corner[:, 0:1, :]) + bbox_hand_corner[:, 0:1, :]
        joints2d = joints2d / self.inp_res - 0.5
        bbox_hand[:, 2:] = bbox_hand[:, 2:] - bbox_hand[:, :2]
        bbox_hand[:, 3] = bbox_hand[:, 2]

        hand_pose = mano_param[:, 3:48]
        global_orient = mano_param[:, :3]
        for idx in range(mano_side.shape[0]):
            cur_pose = hand_pose[idx:idx + 1]
            cur_root = global_orient[idx:idx + 1]
            if mano_side[idx].item() == 0:
                cur_pose = np.dot(cur_pose, self.dex_mano_right.th_selected_comps.numpy())
            elif mano_side[idx].item() == 1:
                cur_pose = np.dot(cur_pose, self.dex_mano_left.th_selected_comps.numpy())
                cur_pose = cur_pose.reshape(15, 3)
                cur_root[:, 1:3] = - cur_root[:, 1:3]
                cur_pose[:, 1:3] = - cur_pose[:, 1:3]
                cur_pose = cur_pose.reshape(-1, 45)

                imgs[idx] = imgs[idx].flip(1)
                imgs_cv[idx] = imgs_cv[idx].flip(1)
                joints2d[idx][:, 0] = -joints2d[idx][:, 0]
            else:
                assert False, f"mano side:{mano_side[idx].item()}"
            hand_pose[idx:idx + 1] = cur_pose
            global_orient[idx:idx + 1] = cur_root
        mano_param[:, 3:48] = hand_pose
        mano_param[:, :3] = global_orient

        sample["img"] = imgs
        sample["mano_param"] = mano_param
        sample["mano_pose"] = mano_param[:, :48]
        sample["mano_shape"] = mano_param[:, 48:]
        sample["bbox"] = bbox_hand
        sample['joints2d'] = joints2d

        for k in sample.keys():
            sample[k] = torch.FloatTensor(sample[k])

        return sample


class DexYCB_Temporal_Twohand(data.Dataset):
    def __init__(
            self,
            dataset_root,
            setup="s0",
            train_label_root="./dexycb-process",
            mode="train",
            inp_res=256,
            T=8,
    ):

        self.dex_single = DexYCB_Temporal(dataset_root, setup, train_label_root,
                                          mode, inp_res, T)
        self.T = T

    def __len__(self):
        return len(self.dex_single)

    def __getitem__(self, index):
        right_id = index
        left_id = index

        right_sample = self.dex_single[right_id]
        left_sample = self.dex_single[left_id]
        output = {'right': right_sample, 'left': left_sample}
        sample = collections.defaultdict(list)
        for k in right_sample.keys():
            if k == 'mano_side':
                continue
            for hand_side in ['right', 'left']:
                sample[f'{k}_{hand_side}'] = output[hand_side][k]

        for hand_side in ['right', 'left']:
            sample[f'joints3d_world_{hand_side}'] = torch.zeros(self.T, 21, 3)
            sample[f'verts3d_world_{hand_side}'] = torch.zeros(self.T, 778, 3)

        x_r, y_r, s_r = random.randint(0, 150), random.randint(0, 350), random.randint(20, 100)
        x_l, y_l, s_l = random.randint(250, 350), random.randint(0, 350), int(s_r * (random.random() * 0.6 + 0.7))
        bbox_right = torch.FloatTensor([x_r, y_r, s_r, s_r])
        bbox_left = torch.FloatTensor([x_l, y_l, s_l, s_l])

        bbox_full = torch.zeros_like(bbox_left)
        bbox_full[2] = 500
        bbox_full[3] = 500

        sample['bbox_right'] = torch.stack([bbox_right] * self.T, dim=0)
        sample['bbox_left'] = torch.stack([bbox_left] * self.T, dim=0)
        sample['bbox_full'] = torch.stack([bbox_full] * self.T, dim=0)

        sample["img_full"] = torch.zeros(self.T, 384, 384, 3)
        sample['inter'] = torch.zeros(self.T)

        del sample["bbox_hand_right"]
        del sample["bbox_hand_left"]

        return sample
