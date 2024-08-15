from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time
import tqdm
import einops
import pyrender
import trimesh
import pickle
import math
from hands_4d.models import load_from_ckpt as load_base
from hands_4d.utils import recursive_to
from hands_4d.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD, \
    ViTDetInterDataset_Batch, HandToken_Sequence
from hands_4d.utils.renderer import Renderer, cam_crop_to_full
from hands_4d.datasets.vitdet_dataset import ViTDetInterDataset_Sequence
from hands_multiview.models import load_from_ckpt as load_multi

LIGHT_BLUE = (0.1255, 0.8157, 0.9176)
LIGHT_RED = (0.98039216, 0.47058824, 0.49411765)
from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

manoData_R = pickle.load(open('/workspace/hamer_twohand/_DATA/data/mano/MANO_RIGHT.pkl', 'rb'), encoding='latin1')
manoData_L = pickle.load(open('/workspace/hamer_twohand/_DATA/data/mano/MANO_LEFT.pkl', 'rb'), encoding='latin1')
def save_mesh_to_ply(vertex_data, file_path,hand_type='right'):
    if vertex_data.shape[0] == 1:
        vertex_data = vertex_data[0]
    face_data_dict = {'right':manoData_R['f'].astype(np.int64),'left':manoData_L['f'].astype(np.int64)}
    face_data = face_data_dict[hand_type]
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

def simple_bbox(video_path, bbox_save_path):
    videoCapture = cv2.VideoCapture(video_path)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    bboxes_right = {}
    bboxes_left = {}
    for idx in range(int(fNUMS)):
        bboxes_right[idx] = [0, 0, size[0], size[1]]
        bboxes_left[idx] = [0, 0, size[0], size[1]]
    bboxes = {"right": bboxes_right, "left": bboxes_left}
    with open(bbox_save_path, 'w') as f:
        json.dump(bboxes, f)

def detect_video(video_path, bbox_save_path):
    # Load detector
    from hands_4d.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hands_4d
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    cfg_path = Path(hands_4d.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    # keypoint detector
    cpm = ViTPoseModel(device)

    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    proc_bar = tqdm.tqdm(total=fNUMS)
    proc_bar.set_description('Making boxes:')
    success, frame = videoCapture.read()

    bboxes_right = {}
    bboxes_left = {}
    fid = 0
    # Iterate over all images in folder
    while success:
        proc_bar.update(1)
        img_cv2 = frame

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # Use hands based on hand keypoint detections

        for vitposes in vitposes_out[0:1]:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                # bbox = [0,0,size[0],size[1]]
                bboxes_left[fid] = np.array(bbox).tolist()
            elif fid > 0 and (fid - 1) in bboxes_left.keys():
                bboxes_left[fid] = bboxes_left[fid - 1]

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                # bbox = [0, 0, size[0], size[1]]
                bboxes_right[fid] = np.array(bbox).tolist()
            elif fid > 0 and (fid - 1) in bboxes_right.keys():
                bboxes_right[fid] = bboxes_right[fid - 1]

        fid += 1
        success, frame = videoCapture.read()

    for fid in range(int(fNUMS)):
        if fid not in bboxes_right.keys():
            for idx in range(fid + 1, int(fNUMS)):
                if idx in bboxes_right.keys():
                    bboxes_right[fid] = bboxes_right[idx]
                    break
        if fid not in bboxes_left.keys():
            for idx in range(fid + 1, int(fNUMS)):
                if idx in bboxes_left.keys():
                    bboxes_left[fid] = bboxes_left[idx]
                    break
        if fid not in bboxes_right.keys():
            for idx in range(fid - 1, 0, -1):
                if idx in bboxes_right.keys():
                    bboxes_right[fid] = bboxes_right[idx]
                    break
        if fid not in bboxes_left.keys():
            for idx in range(fid - 1, 0, -1):
                if idx in bboxes_left.keys():
                    bboxes_left[fid] = bboxes_left[idx]
                    break

    bboxes = {"right": bboxes_right, "left": bboxes_left}
    with open(bbox_save_path, 'w') as f:
        json.dump(bboxes, f)


def gen_sequences(f_num, seq_len, gap):
    sequences = []
    for fid in range(f_num):
        seq = []
        start = fid - (seq_len // 2) * gap
        for f in range(seq_len):
            cur = start + f * gap
            if cur < 0:
                seq.append(0)
            elif cur >= f_num:
                seq.append(f_num - 1)
            else:
                seq.append(cur)
        sequences.append(seq)
    return np.array(sequences)

def run_model_image(model_cfg, model, image_path, image_save_path, device):
    from hands_4d.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hands_4d
    cfg_path = Path(hands_4d.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(image_save_path, exist_ok=True)
    img_names = [imp for end in ['.jpg', '.png'] for imp in os.listdir(image_path) if end in imp]

    proc_bar = tqdm.tqdm(total=len(img_names))
    proc_bar.set_description('Processing:')
    for img_name in img_names:
        proc_bar.update(1)
        fpath = os.path.join(image_path, img_name)
        img_cv2 = cv2.imread(str(fpath))
        img_fn = img_name[:img_name.find('.')]
        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bbox_right = {}
        bbox_left = {}
        # Use hands based on hand keypoint detections

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                # bbox = [0,0,size[0],size[1]]
                bbox_left['0'] = np.array(bbox)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                # bbox = [0, 0, size[0], size[1]]
                bbox_right['0'] = np.array(bbox)

        if len(bbox_right) == 0:
            bbox_right['0'] = np.array([0., 0., img_cv2.shape[1], img_cv2.shape[0]])
        if len(bbox_left) == 0:
            bbox_left['0'] = np.array([0., 0., img_cv2.shape[1], img_cv2.shape[0]])
        bboxes = {'right': bbox_right, 'left': bbox_left}


        # Run reconstruction on all detected hands
        dataset = ViTDetInterDataset_Batch(model_cfg, [img_cv2], bboxes, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for batch in dataloader:
            for k in batch.keys():
                batch[k] = batch[k].unsqueeze(1)
            batch = recursive_to(batch, device)
            with torch.no_grad():
                output = model(batch)
            # output = output_list[-1]
            img_size = batch['bbox_full'][0, 0, 2:].detach().cpu().numpy()
            verts_left = output['verts3d_world_left'].detach().cpu().numpy()
            verts_right = output['verts3d_world_right'].detach().cpu().numpy()
            cam_right = output['cam_aligned_right'].detach().cpu().numpy()
            cam_left = output['cam_aligned_left']
            cam_left = cam_left.detach().cpu().numpy()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

            all_verts = [verts_right[0], verts_left[0]]
            all_cam_t = [cam_right[0], cam_left[0]]
            all_right = [np.array([1], dtype=np.float32), np.array([0], dtype=np.float32)]

            render_res = img_size
            py_renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                                     viewport_height=render_res[1],
                                                     point_size=1.0)
            misc_args = dict(
                mesh_base_color=[LIGHT_BLUE, LIGHT_RED],
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                renderer=py_renderer
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size,
                                                     is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            cv2.imwrite(os.path.join(image_save_path, f'{img_fn}_output.jpg'), 255 * input_img_overlay[:, :, ::-1])

            py_renderer.delete()

def run_model_multi(model_cfg, model, image_path, image_save_path, device):
    from hands_4d.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hands_4d
    cfg_path = Path(hands_4d.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(image_save_path, exist_ok=True)
    img_names = [imp for end in ['.jpg', '.png'] for imp in os.listdir(image_path) if end in imp]

    proc_bar = tqdm.tqdm(total=len(img_names))
    proc_bar.set_description('Processing:')
    bbox_right = {}
    bbox_left = {}
    idx = 0
    imgs = []
    for img_name in img_names:
        proc_bar.update(1)
        fpath = os.path.join(image_path, img_name)
        img_cv2 = cv2.imread(str(fpath))
        imgs.append(img_cv2)

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        # Use hands based on hand keypoint detections

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                # bbox = [0,0,size[0],size[1]]
                bbox_left[f'{idx}'] = np.array(bbox)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                # bbox = [0, 0, size[0], size[1]]
                bbox_right[f'{idx}'] = np.array(bbox)

        if f'{idx}' not in bbox_right:
            bbox_right[f'{idx}'] = np.array([0., 0., img_cv2.shape[1], img_cv2.shape[0]])
        if f'{idx}' not in bbox_left:
            bbox_left[f'{idx}'] = np.array([0., 0., img_cv2.shape[1], img_cv2.shape[0]])
        idx += 1

    bboxes = {'right': bbox_right, 'left': bbox_left}

    # Run reconstruction on all detected hands
    dataset = ViTDetInterDataset_Batch(model_cfg, imgs, bboxes, rescale_factor=2.0, stack_all=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in dataloader:
        batch = recursive_to(batch, device)
        model.seq_len = batch['bbox_full'].shape[1]
        model.aggregate_head.seq_len = batch['bbox_full'].shape[1]
        with torch.no_grad():
            output = model(batch)
        # output = output_list[-1]
        verts_left = output['verts3d_world_left'].detach().cpu().numpy()
        verts_right = output['verts3d_world_right'].detach().cpu().numpy()
        cam_right = output['cam_aligned_right'].detach().cpu().numpy()
        cam_left = output['cam_aligned_left']
        cam_left = cam_left.detach().cpu().numpy()
        for i in range(batch['bbox_full'].shape[1]):
            img_size = batch['bbox_full'][0, i, 2:].detach().cpu().numpy()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

            all_verts = [verts_right[i], verts_left[i]]
            all_cam_t = [cam_right[i], cam_left[i]]
            all_right = [np.array([1], dtype=np.float32), np.array([0], dtype=np.float32)]

            render_res = img_size
            py_renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                                     viewport_height=render_res[1],
                                                     point_size=1.0)
            misc_args = dict(
                mesh_base_color=[LIGHT_BLUE, LIGHT_RED],
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                renderer=py_renderer
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size,
                                                     is_right=all_right, **misc_args)

            # Overlay image
            input_img = imgs[i].astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            img_fn = img_names[i][:img_names[i].find('.')]
            cv2.imwrite(os.path.join(image_save_path, f'{img_fn}_output.jpg'), 255 * input_img_overlay[:, :, ::-1])

            py_renderer.delete()

def run_model_video(model_cfg, model, video_path, bbox_save_path, video_save_path, smooth=False, device='cuda:0'):
    with open(bbox_save_path, 'r') as f:
        bboxes = json.load(f)

    renderer = Renderer(model_cfg, faces=model.mano.faces)
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fNUMS = int(fNUMS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    vw = cv2.VideoWriter(video_save_path, fourcc, fps, size)

    print("smoothen\n")
    smooth_len = 30
    seqs = gen_sequences(fNUMS, seq_len=smooth_len, gap=1)
    smooth_bboxes = {'right':{}, "left":{}}
    for i in tqdm.tqdm(range(len(seqs))):
        seq = seqs[i]
        scale_right = np.zeros(2).astype(np.float64)
        scale_left = np.zeros(2).astype(np.float64)
        for s in seq:
            scale_right = scale_right + np.array(bboxes["right"][f"{s}"]).astype(np.float64)[2:]
            scale_left = scale_left + np.array(bboxes["left"][f"{s}"]).astype(np.float64)[2:]

        scale_right = scale_right / smooth_len
        scale_left = scale_left / smooth_len

        smooth_bboxes["right"][f"{i}"] = np.concatenate([np.array(bboxes["right"][f"{i}"]).astype(np.float64)[:2], scale_right], axis=0)
        smooth_bboxes["left"][f"{i}"] = np.concatenate([np.array(bboxes["left"][f"{i}"]).astype(np.float64)[:2], scale_left], axis=0)

    bboxes = smooth_bboxes
    imgs = []
    success, frame = videoCapture.read()
    while success:
        img_cv2 = torch.Tensor(frame).cpu()
        imgs.append(img_cv2)
        success, frame = videoCapture.read()
    sequences = gen_sequences(fNUMS, model_cfg.MODEL.SEQ_LEN, 10)
    dataset_single = ViTDetInterDataset_Batch(model_cfg, imgs, bboxes, rescale_factor=2.0)
    dataset_seq = ViTDetInterDataset_Sequence(model_cfg, imgs, bboxes, sequences, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset_single, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    render_res = dataset_single[0]['bbox_full'][2:].cpu().numpy()
    py_renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                             viewport_height=render_res[1],
                                             point_size=1.0)
    verts_right_list, verts_left_list = [], []
    cams_right_list, cams_left_list = [], []
    bbox_right, bbox_left = [], []
    hand_tokens = []
    print("Running Model Token")
    for batch in tqdm.tqdm(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            tokens = model.inference_token_forward(batch)
            tokens = einops.rearrange(tokens, '(b s) c -> b s c', s=2)
            hand_tokens.append(tokens.detach().cpu())
            bbox_right.append(batch['bbox_right'].cpu().numpy())
            bbox_left.append(batch['bbox_left'].cpu().numpy())
    bbox_right = np.concatenate(bbox_right, axis=0)
    bbox_left = np.concatenate(bbox_left, axis=0)
    hand_tokens = torch.cat(hand_tokens, dim=0)
    token_dataset = HandToken_Sequence(hand_tokens, sequences)
    token_loader = torch.utils.data.DataLoader(token_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)
    seq_loader = torch.utils.data.DataLoader(dataset_seq, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

    proc_bar = tqdm.tqdm(total=len(seq_loader))
    proc_bar.set_description('Runing Temp:')
    for tokens, batch in zip(token_loader, seq_loader):
        proc_bar.update(1)
        batch = recursive_to(batch, device)
        tokens = tokens.to(device)  # B x T x S x C
        tokens = einops.rearrange(tokens, 'b t s c -> (b t s) c').to(device)
        with torch.no_grad():
            output = model.inference_temp_forward(tokens, batch)
        pred_vert_right, pred_vert_left = output['verts3d_world_right'], output['verts3d_world_left']
        pred_cam_right, pred_cam_left = output['cam_aligned_right'], output['cam_aligned_left']
        verts_right_list.append(pred_vert_right.detach().cpu().numpy())
        verts_left_list.append(pred_vert_left.detach().cpu().numpy())
        cams_right_list.append(pred_cam_right.detach().cpu().numpy())
        cams_left_list.append(pred_cam_left.detach().cpu().numpy())

    verts_right_list = np.concatenate(verts_right_list, axis=0)
    verts_left_list = np.concatenate(verts_left_list, axis=0)
    cams_right_list = np.concatenate(cams_right_list, axis=0)
    cams_left_list = np.concatenate(cams_left_list, axis=0)
    if smooth:
        verts_right_list, verts_left_list = seq_smooth(verts_right_list, verts_left_list)

    img_size = batch['bbox_full'][0, 0, 2:].detach().cpu().numpy()
    print(f"Rendering ")
    for i in tqdm.tqdm(range(fNUMS)):
        verts_left = verts_left_list[i]
        verts_right = verts_right_list[i]
        cam_t_right = cams_right_list[i]
        cam_t_left = cams_left_list[i]
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        all_verts = [verts_right, verts_left]
        all_cam_t = [cam_t_right, cam_t_left]
        all_right = [np.array([1], dtype=np.float32), np.array([0], dtype=np.float32)]

        misc_args = dict(
            mesh_base_color=[LIGHT_BLUE, LIGHT_RED],
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
            renderer=py_renderer
        )

        cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size,
                                                 is_right=all_right, **misc_args)

        # Overlay image
        img_cv2 = imgs[i].cpu().numpy()
        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
        input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        output_img = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)

        vw.write(output_img)
    py_renderer.delete()


def draw_kp(img, j2d_right, j2d_left):
    bones = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    for j in range(21):
        cv2.circle(img, [int(j2d_left[j, 0]), int(j2d_left[j, 1])], 3, [0, 0, 255], 1)
        cv2.circle(img, [int(j2d_right[j, 0]), int(j2d_right[j, 1])], 3, [0, 255, 0], 1)

    for ja, jb in bones:
        cv2.line(img, [int(j2d_left[ja, 0]), int(j2d_left[ja, 1])], [int(j2d_left[jb, 0]), int(j2d_left[jb, 1])],
                [0, 0, 255], 2)
        cv2.line(img, [int(j2d_right[ja, 0]), int(j2d_right[ja, 1])], [int(j2d_right[jb, 0]), int(j2d_right[jb, 1])],
                [0, 255, 0], 2)
    return img

def rot_mesh(verts_right, verts_left, t_right, t_left):
    rot_front = np.array([
        [0.984807753, 0., -0.1736481777],
        [0., 1., 0.],
        [0.1736481777, 0., 0.984807753]
    ], dtype=verts_right.dtype)

    rot_up = np.array([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.]
    ], dtype=verts_right.dtype)

    rot_side = np.array([
        [0., 0., 1.],
        [0., 1., 0.],
        [-1., 0., 0.]
    ], dtype=verts_right.dtype)

    verts_right = verts_right + t_right[np.newaxis, :]
    verts_left = verts_left + t_left[np.newaxis, :]
    root_cen = (verts_right[70:71] + verts_left[70:71]) / 2
    rooted_right = verts_right - root_cen
    rooted_left = verts_left - root_cen
    up_right = rooted_right.dot(rot_up)
    side_right = rooted_right.dot(rot_side)
    front_right = rooted_right.dot(rot_front)
    up_left = rooted_left.dot(rot_up)
    side_left = rooted_left.dot(rot_side)
    front_left = rooted_left.dot(rot_front)
    up_right = up_right + root_cen
    side_right = side_right + root_cen
    front_right = front_right + root_cen
    up_left = up_left + root_cen
    side_left = side_left + root_cen
    front_left = front_left + root_cen
    return up_right, side_right, front_right, up_left, side_left, front_left

def seq_smooth(verts_right, verts_left):
    last_r, last_l = None, None
    v_r, v_l, a_r, a_l = None, None, None, None
    last_v_r, last_v_l = None, None

    for i in range(verts_right.shape[0]):
        if last_r is not None and v_r is not None and a_r is not None:
            pred_r = last_r + v_r + 0.5 * a_r
            verts_right[i] = verts_right[i] * 0.7 + pred_r * 0.3

        if last_l is not None and v_l is not None and a_l is not None:
            pred_l = last_l + v_l + 0.5 * a_l
            verts_left[i] = verts_left[i] * 0.7 + pred_l * 0.3

        if last_r is not None:
            v_r = verts_right[i] - last_r

        if last_v_r is not None and v_r is not None:
            a_r = v_r - last_v_r

        if last_l is not None:
            v_l = verts_left[i] - last_l

        if last_v_l is not None and v_l is not None:
            a_l = v_l - last_v_l

        last_r = verts_right[i]
        last_v_r = v_r

        last_l = verts_left[i]
        last_v_l = v_l

    return verts_right, verts_left

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--mode', type=str, default='image')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--video_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    mode = args.mode
    assert mode in ['image', 'video', 'multi']
    if mode == 'image':
        model, model_cfg = load_base(args.checkpoint, args.cfg)
    elif mode == 'video':
        model, model_cfg = load_base(args.checkpoint, args.cfg)
    elif mode == 'multi':
        model, model_cfg = load_multi(args.checkpoint, args.cfg)
    # Setup HaMeR model
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    image_input_path = args.image_dir
    video_input_path = args.video_dir
    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)

    if mode == 'video':
        vf = video_input_path.split('/')[-1]
        vname = vf[:vf.find('.mp4')]
        bbox_path = os.path.join(output_path, f'{vname}/bbox.json')
        video_save_path = os.path.join(output_path, f'{mode}_{vname}.mp4')

        if not os.path.exists(bbox_path):
            detect_video(video_input_path, bbox_path)

    if mode == 'image':
        run_model_image(model_cfg, model, image_input_path, output_path, device=device)
    elif mode == 'video':
        run_model_video(model_cfg, model, video_input_path, bbox_path, video_save_path, smooth=True, device=device)
    elif mode == 'multi':
        run_model_multi(model_cfg, model, image_input_path, output_path, device=device)

if __name__ == '__main__':
    main()
