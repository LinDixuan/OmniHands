from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None) # BBOX_SHAPE 192 x 256
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        right = self.right[idx].copy()
        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)   #patch width: 256

            #print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv) # 3 x 256 x 256   0~255
        img_patch_copy = img_patch.copy()
        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
            'cv_img':img_patch_copy,
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = self.right[idx].copy()
        return item

class ViTDetInterDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.int32)
        if right.shape[0] == 2 and np.sum(right) == 1:
            self.length = 1
        else:
            self.length = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, np.array]:
        output = {}
        for idx in range(2):

            center = self.center[idx].copy()
            center_x = center[0]
            center_y = center[1]

            scale = self.scale[idx]
            BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None) # BBOX_SHAPE 192 x 256
            bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

            patch_width = patch_height = self.img_size

            right = self.right[idx].copy()
            flip = right == 0
            if right == 0:
                side = 'left'
            else:
                side = 'right'

            # 3. generate image patch
            # if use_skimage_antialias:
            cvimg = self.img_cv2.copy()
            if True:
                # Blur image to avoid aliasing artifacts
                downsampling_factor = ((bbox_size*1.0) / patch_width)   #patch width: 256

                #print(f'{downsampling_factor=}')
                downsampling_factor = downsampling_factor / 2.0
                if downsampling_factor > 1.1:
                    cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


            img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                        center_x, center_y,
                                                        bbox_size, bbox_size,
                                                        patch_width, patch_height,
                                                        flip, 1.0, 0,
                                                        border_mode=cv2.BORDER_CONSTANT)
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = convert_cvimg_to_tensor(img_patch_cv) # 3 x 256 x 256   0~255
            img_patch_copy = img_patch.copy()
            # apply normalization
            for n_c in range(min(self.img_cv2.shape[2], 3)):
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

            item = {
                'img': img_patch,
                'personid': int(self.personid[idx]),
                'cv_img':img_patch_copy,
            }
            item['box_center'] = self.center[idx].copy()
            item['box_size'] = bbox_size
            item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
            item['right'] = self.right[idx].copy()
            bbox = np.array([center[0] - bbox_size / 2, center[1] - bbox_size / 2, bbox_size, bbox_size])
            output[f'img_{side}'] = torch.from_numpy(img_patch).permute(1,2,0).float()
            output[f'cv_img_{side}'] = img_patch_copy
            output[f'bbox_{side}'] = bbox
        output['bbox_full'] = np.asarray([0.,0.,self.img_cv2.shape[1],self.img_cv2.shape[0]])
        output['right'] = self.right.astype(np.float32)
        for k in output.keys():
            if isinstance(output[k],np.ndarray):
                output[k] = torch.from_numpy(output[k]).float()
        return output


class ViTDetInterDataset_Batch(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 imgs: list,
                 bboxes: Dict,
                 rescale_factor=2.5,
                 offset=0
                 ):
        super().__init__()
        self.cfg = cfg
        self.imgs = imgs
        self.bboxes = bboxes
        self.rescale_factor = rescale_factor
        self.offset = offset

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.right = np.array([1, 0])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> Dict[str, np.array]:
        img = self.imgs[index].cpu().numpy()
        # Preprocess annotations
        box_right = np.array(self.bboxes["right"][f"{index+self.offset}"]).astype(np.float32)
        box_left = np.array(self.bboxes["left"][f"{index+self.offset}"]).astype(np.float32)
        boxes = np.stack([box_right, box_left], axis=0)

        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

        output = {}
        for idx in range(2):

            center = centers[idx].copy()
            center_x = center[0]
            center_y = center[1]

            scale = scales[idx]
            BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None) # BBOX_SHAPE 192 x 256
            bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

            patch_width = patch_height = self.img_size

            right = self.right[idx].copy()
            flip = right == 0
            if right == 0:
                side = 'left'
            else:
                side = 'right'

            # 3. generate image patch
            # if use_skimage_antialias:
            cvimg = img.copy()
            if True:
                # Blur image to avoid aliasing artifacts
                downsampling_factor = ((bbox_size*1.0) / patch_width)   #patch width: 256

                #print(f'{downsampling_factor=}')
                downsampling_factor = downsampling_factor / 2.0
                if downsampling_factor > 1.1:
                    cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


            img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                        center_x, center_y,
                                                        bbox_size, bbox_size,
                                                        patch_width, patch_height,
                                                        flip, 1.0, 0,
                                                        border_mode=cv2.BORDER_CONSTANT)
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = convert_cvimg_to_tensor(img_patch_cv)   # 3 x 256 x 256   0~255
            img_patch_copy = img_patch.copy()
            # apply normalization
            for n_c in range(min(img.shape[2], 3)):
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

            item = {
                'img': img_patch,
                'cv_img':img_patch_copy,
            }
            item['box_center'] = centers[idx].copy()
            item['box_size'] = bbox_size
            item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
            item['right'] = self.right[idx].copy()
            bbox = np.array([center[0] - bbox_size / 2, center[1] - bbox_size / 2, bbox_size, bbox_size])
            output[f'img_{side}'] = torch.from_numpy(img_patch).permute(1, 2, 0).float()
            output[f'cv_img_{side}'] = img_patch_copy
            output[f'bbox_{side}'] = bbox
            output[f'box_center_{side}'] = centers[idx].copy()
            output[f'box_size_{side}'] = bbox_size

        bbox_scale_left = output['box_size_left']
        bbox_scale_right = output['box_size_right']

        """bbox_scale = max(bbox_scale_left, bbox_scale_right)

        output['box_size_left'] = bbox_scale
        output['box_size_right'] = bbox_scale"""
        output['bbox_left'] = np.array([output['box_center_left'][0] - bbox_scale_left / 2,
                                        output['box_center_left'][1] - bbox_scale_left / 2,
                                        bbox_scale_left, bbox_scale_left])
        output['bbox_right'] = np.array([output['box_center_right'][0] - bbox_scale_right / 2,
                                        output['box_center_right'][1] - bbox_scale_right / 2,
                                        bbox_scale_right, bbox_scale_right])

        output['bbox_full'] = np.asarray([0., 0., img.shape[1], img.shape[0]])
        output['right'] = self.right.astype(np.float32)
        for k in output.keys():
            if isinstance(output[k],np.ndarray):
                output[k] = torch.from_numpy(output[k]).float()
        return output

class ViTDetInterDataset_Sequence(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 imgs: list,
                 bboxes: Dict,
                 sequences: list,
                 rescale_factor=2.5,
                 offset=0
                 ):
        super().__init__()
        self.cfg = cfg
        self.imgs = imgs
        self.bboxes = bboxes
        self.rescale_factor = rescale_factor
        self.offset = offset
        self.sequences = sequences

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.right = np.array([1, 0])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, fid: list[int]) -> Dict[str, np.array]:
        index_list = self.sequences[fid]
        seq_output = []
        for index in index_list:
            img = self.imgs[index].cpu().numpy()
            # Preprocess annotations
            box_right = np.array(self.bboxes["right"][f"{index+self.offset}"]).astype(np.float32)
            box_left = np.array(self.bboxes["left"][f"{index+self.offset}"]).astype(np.float32)
            boxes = np.stack([box_right, box_left], axis=0)

            centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
            scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

            output = {}
            for idx in range(2):

                center = centers[idx].copy()
                center_x = center[0]
                center_y = center[1]

                scale = scales[idx]
                BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None) # BBOX_SHAPE 192 x 256
                bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

                patch_width = patch_height = self.img_size

                right = self.right[idx].copy()
                flip = right == 0
                if right == 0:
                    side = 'left'
                else:
                    side = 'right'

                # 3. generate image patch
                # if use_skimage_antialias:
                cvimg = img.copy()
                if True:
                    # Blur image to avoid aliasing artifacts
                    downsampling_factor = ((bbox_size*1.0) / patch_width)   #patch width: 256

                    #print(f'{downsampling_factor=}')
                    downsampling_factor = downsampling_factor / 2.0
                    if downsampling_factor > 1.1:
                        cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


                img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                            center_x, center_y,
                                                            bbox_size, bbox_size,
                                                            patch_width, patch_height,
                                                            flip, 1.0, 0,
                                                            border_mode=cv2.BORDER_CONSTANT)
                img_patch_cv = img_patch_cv[:, :, ::-1]
                img_patch = convert_cvimg_to_tensor(img_patch_cv)   # 3 x 256 x 256   0~255
                img_patch_copy = img_patch.copy()
                # apply normalization
                for n_c in range(min(img.shape[2], 3)):
                    img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

                item = {
                    'img': img_patch,
                    'cv_img':img_patch_copy,
                }
                item['box_center'] = centers[idx].copy()
                item['box_size'] = bbox_size
                item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
                item['right'] = self.right[idx].copy()
                bbox = np.array([center[0] - bbox_size / 2, center[1] - bbox_size / 2, bbox_size, bbox_size])
                output[f'img_{side}'] = torch.from_numpy(img_patch).permute(1,2,0).float()
                output[f'bbox_{side}'] = bbox
                output[f'box_center_{side}'] = centers[idx].copy()
                output[f'box_size_{side}'] = bbox_size

            bbox_scale_left = output['box_size_left']
            bbox_scale_right = output['box_size_right']

            output['bbox_left'] = np.array([output['box_center_left'][0] - bbox_scale_left / 2,
                                            output['box_center_left'][1] - bbox_scale_left / 2,
                                            bbox_scale_left, bbox_scale_left])
            output['bbox_right'] = np.array([output['box_center_right'][0] - bbox_scale_right / 2,
                                             output['box_center_right'][1] - bbox_scale_right / 2,
                                             bbox_scale_right, bbox_scale_right])

            output['bbox_full'] = np.asarray([0., 0., img.shape[1], img.shape[0]])
            del output['box_size_left']
            del output['box_size_right']
            for k in output.keys():
                output[k] = torch.FloatTensor(output[k])
            seq_output.append(output)

        batch = {}
        for k in seq_output[0].keys():

            batch[k] = []
        for idx in range(len(seq_output)):
            info = seq_output[idx]
            for k in batch.keys():
                batch[k].append(info[k])
        for k in batch.keys():
            batch[k] = torch.stack(batch[k], dim=0)
        return batch

class HandToken_Sequence(torch.utils.data.Dataset):
    def __init__(self,
                 hand_tokens,
                 sequences):
        self.hand_tokens = hand_tokens
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        output_tokens = []
        for fid in seq:
            output_tokens.append(self.hand_tokens[fid:fid+1])

        output_tokens = torch.cat(output_tokens, dim=0)

        return output_tokens