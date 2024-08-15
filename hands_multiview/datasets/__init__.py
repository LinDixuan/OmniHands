from typing import Dict, Optional

import torch
import numpy as np
import pytorch_lightning as pl
from yacs.config import CfgNode
import random
import cv2
import os
import collections

import webdataset as wds
from ..configs import to_lower
from .dataset import Dataset
from .image_dataset import ImageDataset
from .mocap_dataset import MoCapDataset
from .arctic_multi import Arctic_Multiview
from .inter_multi import get_multi_dataset
from .relighted_multi import Relighted_Multiview
from .relighted_mano import MANO
from ..utils.geometry import aa_to_rotmat, perspective_projection
from .interhand_temp import get_temp_dataset
from .dexycb_multi import DexYCB_Temporal_Twohand

def create_webdataset(cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> Dataset:
    """
    Like `create_dataset` but load data from tars.
    """
    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type.load_tars_as_webdataset(cfg, **to_lower(dataset_cfg), train=train)


class MixedWebDataset(wds.WebDataset):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> None:
        super(wds.WebDataset, self).__init__()
        dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        datasets = [create_webdataset(cfg, dataset_cfg[dataset], train=train) for dataset, v in dataset_list.items()]
        weights = np.array([v.WEIGHT for dataset, v in dataset_list.items()])
        weights = weights / weights.sum()  # normalize
        self.append(wds.RandomMix(datasets, weights))

class HAMERDataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for HAMER training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mocap_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        if self.train_dataset == None:
            self.train_dataset = MixedWebDataset(self.cfg, self.dataset_cfg, train=True).with_epoch(100_000).shuffle(4000)
            self.val_dataset = MixedWebDataset(self.cfg, self.dataset_cfg, train=False).shuffle(4000)
            self.mocap_dataset = MoCapDataset(**to_lower(self.dataset_cfg[self.cfg.DATASETS.MOCAP]))


    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS, prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR)
        mocap_dataloader = torch.utils.data.DataLoader(self.mocap_dataset, self.cfg.TRAIN.NUM_TRAIN_SAMPLES * self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=1)
        return {'img': train_dataloader, 'mocap': mocap_dataloader}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader  
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return val_dataloader

class hamer_filtered_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_cfg, T, task='train'):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.task = task
        self.hamer_datamodule = HAMERDataModule(cfg, dataset_cfg)
        self.hamer_datamodule.setup()
        self.train_dataset = self.hamer_datamodule.train_dataset
        self.train_iter = self.train_dataset.__iter__()
        self.val_dataset = self.hamer_datamodule.val_dataset
        self.val_iter = self.val_dataset.__iter__()
        self.seq_len = T
        self.mano = MANO()

        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)

    def __len__(self):
        if self.task == 'train':
            return 200_000
        else:
            return 200_000

    def get_item(self, iterator):
        while True:
            ts = next(iterator)
            vl = 1
            for k in ts['has_mano_params'].keys():
                if ts['has_mano_params'][k] < 0.5:
                    vl = 0
            df = 1
            if ts['mano_params_is_axis_angle']['global_orient'] and ts['mano_params_is_axis_angle']['hand_pose']:
                df = 0
            if vl == 1 and df == 0:
                return ts

    def __getitem__(self, index):

        if self.task == 'train':
            data_right = self.get_item(self.train_iter)
            data_left = self.get_item(self.train_iter)
        else:
            data_right = self.get_item(self.val_iter)
            data_left = self.get_item(self.val_iter)
        sample = collections.defaultdict(list)
        data_hm = {'left': data_left, 'right': data_right}
        data = collections.defaultdict(list)

        for hand_side in ['right', 'left']:
            img = torch.FloatTensor(data_hm[hand_side]['img'].copy()).permute(1, 2, 0).flip(2)
            for n_c in range(3):
                img[n_c, :, :] = img[n_c, :, :] * self.STD[n_c] + self.MEAN[n_c]
            data[f'img_{hand_side}'] = img
            hand_pose = torch.FloatTensor(data_hm[hand_side]['mano_params']['hand_pose']).reshape(45)
            global_orient = torch.FloatTensor(data_hm[hand_side]['mano_params']['global_orient']).reshape(3)
            data[f'mano_pose_{hand_side}'] = torch.cat([global_orient, hand_pose], dim=0) \
                                - torch.FloatTensor(self.mano.layer["right"].pose_mean)
            data[f'mano_shape_{hand_side}'] = torch.FloatTensor(data_hm[hand_side]['mano_params']['betas'])
            data[f'mano_param_{hand_side}'] = torch.cat([data[f'mano_pose_{hand_side}'],
                                                         data[f'mano_shape_{hand_side}']], dim=0)

            data[f'joints2d_{hand_side}'] = torch.FloatTensor(data_hm[hand_side]['keypoints_2d'][:, :-1])

        x_r, y_r, s_r = random.randint(0, 150), random.randint(0, 350), random.randint(20, 100)
        x_l, y_l, s_l = random.randint(250, 350), random.randint(0, 350), int(s_r * (random.random() * 0.6 + 0.7))
        bbox_right = torch.FloatTensor([x_r, y_r, s_r, s_r])
        bbox_left = torch.FloatTensor([x_l, y_l, s_l, s_l])
        bbox_full = torch.zeros_like(bbox_left)
        bbox_full[2] = 500
        bbox_full[3] = 500

        data['bbox_right'] = bbox_right
        data['bbox_left'] = bbox_left
        data['bbox_full'] = bbox_full
        data['img_full'] = torch.zeros(384, 384, 3)

        data[f'joints3d_world_right'] = torch.zeros(21, 3)
        data[f'verts3d_world_right'] = torch.zeros(778, 3)
        data[f'joints3d_world_left'] = torch.ones(21, 3)
        data[f'verts3d_world_left'] = torch.ones(778, 3)

        for k in data.keys():
            if isinstance(data[k], np.ndarray):
                data[k] = torch.from_numpy(data[k])
        for k in data.keys():
            sample[k] = torch.stack([data[k]] * self.seq_len, dim=0)
        sample['inter'] = torch.zeros(self.seq_len)
        sample['dataset'] = torch.zeros(self.seq_len)

        return sample


class mix_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_list,
                 dataset_prob):
        super().__init__()
        self.dataset_list = dataset_list
        self.dataset_prob = dataset_prob
        self.dataset_num = len(self.dataset_list)
        assert len(dataset_prob) == 0 or len(dataset_prob) == self.dataset_num
        if len(dataset_prob) == 0:
            self.dataset_prob = [1 for _ in range(self.dataset_num)]

        self.dataset_prob = np.array(self.dataset_prob)
        self.dataset_prob = self.dataset_prob / self.dataset_prob.sum()
        self.dataset_prob = np.cumsum(self.dataset_prob)

        self.size = 0
        for i in range(self.dataset_num):
            self.size = self.size + len(self.dataset_list[i])

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        prob = random.random()
        dataset_id = np.searchsorted(self.dataset_prob, prob, side='right')

        dataset = self.dataset_list[dataset_id]
        idx = random.randint(0, len(dataset) - 1)
        data = dataset[idx]
        j2d_r, j2d_l = data['joints2d_right'].clone(), data['joints2d_left'].clone()
        bbox_r, bbox_l, bbox_f = data['bbox_right'], data['bbox_left'], data['bbox_full']
        j2d_l[..., 0] = -j2d_l[..., 0]
        j2d_r = (j2d_r + 0.5) * bbox_r[:, 2:4].unsqueeze(1) + bbox_r[:, :2].unsqueeze(1)
        j2d_l = (j2d_l + 0.5) * bbox_l[:, 2:4].unsqueeze(1) + bbox_l[:, :2].unsqueeze(1)
        j2d_r = (j2d_r - bbox_f[:, :2].unsqueeze(1)) / bbox_f[:, 2:4].unsqueeze(1) - 0.5
        j2d_l = (j2d_l - bbox_f[:, :2].unsqueeze(1)) / bbox_f[:, 2:4].unsqueeze(1) - 0.5

        data['joints2d_glb_right'] = j2d_r
        data['joints2d_glb_left'] = j2d_l
        return data


class Mix_multi(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for HAMER training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        dex_data_root = "/workspace/twohand_full/Dexycb"
        dex_label_root = "/workspace/twohand_full/Dexycb/dexycb-process"
        #arctic_train = Arctic_Multiview(T=cfg.MODEL.SEQ_LEN)
        inter26_train = get_multi_dataset(T=cfg.MODEL.SEQ_LEN, split="train")
        relighted_train = Relighted_Multiview(T=cfg.MODEL.SEQ_LEN)
        inter26_val = get_multi_dataset(T=cfg.MODEL.SEQ_LEN, split="test")
        #arctic_val = Arctic_Multiview(T=cfg.MODEL.SEQ_LEN, split="val")
        #dex_train = DexYCB_Temporal_Twohand(dataset_root=dex_data_root, train_label_root=dex_label_root,
        #                                    mode="train", T=cfg.MODEL.SEQ_LEN)
        #dex_test = DexYCB_Temporal_Twohand(dataset_root=dex_data_root, train_label_root=dex_label_root,
        #                                    mode="test", T=cfg.MODEL.SEQ_LEN)

        self.train_dataset = mix_dataset(dataset_list=[inter26_train],
                                         dataset_prob=[0.4, ])

        self.val_dataset = mix_dataset(dataset_list=[inter26_val],
                                         dataset_prob=[1.])
        #self.val_dataset = inter26_val

        self.test_dataset = inter26_val

        print(f'train: {len(self.train_dataset)}')
        print(f'val: {len(self.val_dataset)}')
        print(f'test: {len(self.test_dataset)}')


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE,
                                                       drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS,
                                                       prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR,
                                                       shuffle=False)

        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        if self.cfg.TRAIN.MULTITHREAD:
            val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 4, drop_last=True, num_workers=8,
                                                         shuffle=False)
        else:
            val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 4, drop_last=True, num_workers=0, shuffle=False)
        return val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:

        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=4, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return test_dataloader

    