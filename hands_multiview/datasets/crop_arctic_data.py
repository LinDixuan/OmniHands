# Copyright (c) Facebook, Inc. and its affiliates.

import os
import numpy as np
import cv2
import json
from glob import glob
import os.path as osp

os.environ["PYOPENGL_PLATFORM"] = "egl"
import smplx
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    DirectionalLights,
    PerspectiveCameras,
    Materials,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex)
import pickle
from tqdm import tqdm
import sys

#sys.path.append('/home/lizipeng/hand_gaussian_avatar/code')
#from toolkit.data_utils import cut_img


def render_mesh(mesh, face, cam_param, render_shape, hand_type):
    batch_size, vertex_num = mesh.shape[:2]
    # mesh = mesh / 1000 # milimeter to meter

    textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().cuda())
    mesh = torch.stack((-mesh[:, :, 0], -mesh[:, :, 1], mesh[:, :, 2]),
                       2)  # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                 principal_point=cam_param['princpt'],
                                 device='cuda',
                                 in_ndc=False,
                                 image_size=torch.LongTensor(render_shape).cuda().view(1, 2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1,
                                            perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
        device='cuda',
        specular_color=color,
        shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:, :, :, :3] * 255
        mask = ~torch.all(images == 255, dim=-1)
        mask = mask.int() * 255
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        # np.save('mask.npy', mask)
        # cv2.imwrite('mask.png', mask.astype(np.uint8))
        depthmaps = fragments.zbuf

    return images, depthmaps, mask


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


# mano layer
smplx_path = {'right': '/workspace/hamer_intertime/_DATA/data/mano/MANO_RIGHT.pkl',
              'left': '/workspace/hamer_intertime/_DATA/data/mano/MANO_LEFT.pkl'}
mano_layer = {'right': smplx.create(smplx_path['right'], 'mano', use_pca=False, is_rhand=True),
              'left': smplx.create(smplx_path['left'], 'mano', use_pca=False, is_rhand=False)}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:, 0, :] *= -1

root_path = '/workspace/twohand_full/arctic/arctic_data/data'
img_root_path = '/workspace/arctic/outputs/decroppped_images'
split_json_path = '/workspace/twohand_full/arctic/arctic_data/data/splits_json/protocol_p1.json'
gt_test_path = '/workspace/hamer_intertime/gt_test/arctic'
os.makedirs(gt_test_path, exist_ok=True)

with open(split_json_path, 'r') as f:
    split_seq = json.load(f)

seq_names = []
for nam in split_seq['val']:
    seq_names.append(nam.split('/')[-1])

annot_root_path = osp.join(root_path, 'raw_seqs')
split = 'val'
sid = 's05'

save_path = os.path.join('/workspace/twohand_full/arctic', 'arctic1')
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)

for seq_name in seq_names:
    # with open(osp.join(annot_root_path, sid, f'{seq_name}.mano.npy')) as f:
    mano_params = np.load(osp.join(annot_root_path, sid, f'{seq_name}.mano.npy'), allow_pickle=True).item()
    bbox_params = np.load(osp.join('/workspace/arctic/outputs/processed/seqs/s05',
                                   f'{seq_name}.npy'), allow_pickle=True).item()


    with open(osp.join(root_path, 'meta', 'misc.json')) as f:
        cam_params = json.load(f)

    intris_mats = {}
    for view_idx in range(9):
        if view_idx == 0:
            continue
        fnames = glob(
            f"{img_root_path}/{sid}/{seq_name}/{view_idx}/*.jpg"
        )
        fnames = sorted(fnames)

        pbar = tqdm(fnames)
        for fname in pbar:
            pbar.set_description(fname)
            vidx = int(osp.basename(fname).split(".")[0]) - cam_params[sid]["ioi_offset"]

            img = cv2.imread(fname)
            for hand_type in ['right', 'left']:
                mano_param = mano_params[hand_type]
                root_pose = torch.FloatTensor(mano_param['rot'][vidx]).view(1, 3)
                hand_pose = torch.FloatTensor(mano_param['pose'][vidx]).view(1, -1)
                shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
                trans = torch.FloatTensor(mano_param['trans'][vidx]).view(1, 3)
                output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)

                verts = output.vertices
                world2cam = torch.FloatTensor(np.array(cam_params[sid]["world2cam"][view_idx - 1])).unsqueeze(0)
                verts_cam = transform_points_batch(world2cam, verts)
                intris_mat = torch.FloatTensor(np.array(cam_params[sid]["intris_mat"][view_idx - 1])).unsqueeze(0)
                verts_2d = project2d_batch(intris_mat, verts_cam)

                cam_K = intris_mat[0].detach().cpu().numpy().copy()
                verts2d = verts_2d[0].detach().cpu().numpy().copy()
            


    with open(os.path.join(save_path, 'intris_mats.pkl'), 'wb') as f:
        pickle.dump(intris_mats, f)
