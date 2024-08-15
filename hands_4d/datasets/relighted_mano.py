# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import torch
import os.path as osp
import cv2
import smplx
def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]

    return np.stack((x,y,z),1)

def cam2pixel_batch(cam_coord, f, c):
    x = cam_coord[:,:,0] / cam_coord[:,:,2] * f[...,0:1] + c[...,0:1]
    y = cam_coord[:,:,1] / cam_coord[:,:,2] * f[...,1:2] + c[...,1:2]
    z = cam_coord[:,:,2]
    return torch.stack((x,y,z),dim=-1)

class MANO(object):
    def __init__(self,use_pca=False):
        human_model_path = '/workspace/hamer_twohand/_DATA/data/'
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False,
                          'create_transl': False, 'num_pca_comps': 45}
        self.layer = {
            'right': smplx.create(human_model_path, 'mano', is_rhand=True, use_pca=use_pca, flat_hand_mean=False,
                                  **self.layer_arg),
            'left': smplx.create(human_model_path, 'mano', is_rhand=False, use_pca=use_pca, flat_hand_mean=False,
                                 **self.layer_arg)}

        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(self.layer['left'].shapedirs[:, 0, :] - self.layer['right'].shapedirs[:, 0, :])) < 1:
            #print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:, 0, :] *= -1

        # original MANO joint set
        self.orig_joint_num = 16
        self.orig_joints_name = (
        'Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3',
        'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.orig_root_joint_idx = self.orig_joints_name.index('Wrist')
        self.orig_flip_pairs = ()
        self.orig_joint_regressor = self.layer['right'].J_regressor.numpy()  # same for the right and left hands

        # changed MANO joint set (single hands)
        self.sh_joint_num = 21  # manually added fingertips
        self.sh_joints_name = (
        'Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1',
        'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3',
        'Pinky_4')
        self.sh_skeleton = (
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13),
        (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))
        self.sh_root_joint_idx = self.sh_joints_name.index('Wrist')
        self.sh_flip_pairs = ()
        # add fingertips to joint_regressor
        self.sh_joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name,
                                                              self.sh_joints_name)
        self.sh_joint_regressor[self.sh_joints_name.index('Thumb_4')] = np.array(
            [1 if i == 745 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.sh_joint_regressor[self.sh_joints_name.index('Index_4')] = np.array(
            [1 if i == 317 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.sh_joint_regressor[self.sh_joints_name.index('Middle_4')] = np.array(
            [1 if i == 445 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.sh_joint_regressor[self.sh_joints_name.index('Ring_4')] = np.array(
            [1 if i == 556 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.sh_joint_regressor[self.sh_joints_name.index('Pinky_4')] = np.array(
            [1 if i == 673 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)

        # changed MANO joint set (two hands)
        self.th_joint_num = 42  # manually added fingertips. two hands
        self.th_joints_name = (
        'R_Wrist', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3',
        'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3',
        'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', 'L_Wrist', 'L_Thumb_1', 'L_Thumb_2',
        'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2',
        'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2',
        'L_Pinky_3', 'L_Pinky_4')
        self.th_root_joint_idx = {'right': self.th_joints_name.index('R_Wrist'),
                                  'left': self.th_joints_name.index('L_Wrist')}
        self.th_flip_pairs = [(i, i + 21) for i in range(21)]
        self.th_joint_type = {'right': np.arange(0, self.th_joint_num // 2),
                              'left': np.arange(self.th_joint_num // 2, self.th_joint_num)}



def get_mano_data(mano_param, cam_param, do_flip, img_shape, mano):

    pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
    hand_type = mano_param['hand_type']
    pose = torch.FloatTensor(pose).view(-1, 3);
    shape = torch.FloatTensor(shape).view(1, -1);  # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

    if do_flip:
        if hand_type == 'right':
            hand_type = 'left'
        else:
            hand_type = 'right'

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3)
        root_pose = pose[mano.orig_root_joint_idx, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in mano.orig_flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
        trans[:, 0] *= -1  # multiply -1

    # get root joint coordinate
    root_pose = pose[mano.orig_root_joint_idx].view(1, 3)
    hand_pose = torch.cat((pose[:mano.orig_root_joint_idx, :], pose[mano.orig_root_joint_idx + 1:, :])).view(1, -1)
    #root_pose, _ = cv2.Rodrigues(np.eye(3))
    #root_pose = torch.from_numpy(root_pose).view(1,3).float()
    #trans = torch.zeros(1,3)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
    mesh_coord = output.vertices[0].numpy()

    joint_coord = np.dot(mano.sh_joint_regressor, mesh_coord)

    # bring geometry to the original (before flip) position
    if do_flip:
        flip_trans_x = joint_coord[mano.sh_root_joint_idx, 0] * -2
        mesh_coord[:, 0] += flip_trans_x
        joint_coord[:, 0] += flip_trans_x

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)

    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                  dtype=np.float32).reshape(1, 3)

        root_coord = joint_coord[mano.sh_root_joint_idx, None, :].copy()
        joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1, 0)).transpose(1, 0) + t
        mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1, 0)).transpose(1, 0) + t


    # flip translation
    if do_flip:  # avg of old and new root joint should be image center.
        focal, princpt = cam_param['focal'], cam_param['princpt']
        flip_trans_x = 2 * (
                    ((img_shape[1] - 1) / 2. - princpt[0]) / focal[0] * joint_coord[mano.sh_root_joint_idx, 2]) - 2 * \
                       joint_coord[mano.sh_root_joint_idx][0]
        mesh_coord[:, 0] += flip_trans_x
        joint_coord[:, 0] += flip_trans_x
    # image projection
    mesh_cam = mesh_coord  # camera-centered 3D coordinates (not root-relative)
    joint_cam = joint_coord  # camera-centered 3D coordinates (not root-relative)
    mesh_img = cam2pixel(mesh_cam,cam_param['focal'], cam_param['princpt'])
    joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = joint_img[:, :2]

    mesh_img = mesh_img[:, :2]
    pose = pose.numpy().reshape(-1)
    shape = shape.numpy().reshape(-1)
    return joint_img, joint_cam, mesh_img, mesh_cam, pose, shape


def get_mano_data_old(mano_param, cam_param, do_flip, img_shape, mano):
    pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
    hand_type = mano_param['hand_type']
    pose = torch.FloatTensor(pose).view(-1, 3);
    shape = torch.FloatTensor(shape).view(1, -1);  # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

    if do_flip:
        if hand_type == 'right':
            hand_type = 'left'
        else:
            hand_type = 'right'

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3)
        root_pose = pose[mano.orig_root_joint_idx, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        new_R = np.dot(R, root_pose)
        root_pose, _ = cv2.Rodrigues(new_R)
        pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)
        new_R = torch.from_numpy(new_R)


    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in mano.orig_flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
        trans[:, 0] *= -1  # multiply -1

    # get root joint coordinate
    hand_pose = torch.cat((pose[:mano.orig_root_joint_idx, :], pose[mano.orig_root_joint_idx + 1:, :])).view(1, -1)
    zero_pose, _ = cv2.Rodrigues(np.eye(3))
    zero_pose = torch.from_numpy(zero_pose).view(1,3).float()
    zero_trans = torch.zeros(1,3)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=zero_pose, transl=zero_trans)
    mesh_coord = output.vertices[0].numpy()

    joint_coord = np.dot(mano.sh_joint_regressor, mesh_coord)

    # bring geometry to the original (before flip) position
    if do_flip:
        flip_trans_x = joint_coord[mano.sh_root_joint_idx, 0] * -2
        mesh_coord[:, 0] += flip_trans_x
        joint_coord[:, 0] += flip_trans_x

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)

    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                  dtype=np.float32).reshape(1, 3)
        R = torch.from_numpy(R)
        t = torch.from_numpy(t)

        t = t + torch.matmul(R, trans.transpose(0,1)).squeeze(-1)

        root_coord = joint_coord[mano.sh_root_joint_idx, None, :].copy()
        root_coord = torch.from_numpy(root_coord)
        mesh_coord = torch.from_numpy(mesh_coord) - root_coord
        joint_coord = torch.from_numpy(joint_coord) - root_coord

        mesh_coord = torch.matmul(mesh_coord,new_R.transpose(0,1)) + t.unsqueeze(0)
        mesh_coord = mesh_coord[0].numpy()
        joint_coord = torch.matmul(joint_coord, new_R.transpose(0, 1)) + t.unsqueeze(0)
        joint_coord = joint_coord[0].numpy()
    # flip translation
    if do_flip:  # avg of old and new root joint should be image center.
        focal, princpt = cam_param['focal'], cam_param['princpt']
        flip_trans_x = 2 * (
                ((img_shape[1] - 1) / 2. - princpt[0]) / focal[0] * joint_coord[mano.sh_root_joint_idx, 2]) - 2 * \
                       joint_coord[mano.sh_root_joint_idx][0]
        mesh_coord[:, 0] += flip_trans_x
        joint_coord[:, 0] += flip_trans_x
    # image projection
    mesh_cam = mesh_coord  # camera-centered 3D coordinates (not root-relative)
    joint_cam = joint_coord  # camera-centered 3D coordinates (not root-relative)

    mesh_img = cam2pixel(mesh_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = joint_img[:, :2]
    mesh_img = mesh_img[:, :2]
    if 'R' in cam_param:
        root = pose[mano.orig_root_joint_idx].view(1,3)
    else:
        root = zero_pose
    root = batch_rodrigues(root).view(1,3,3)
    pose = batch_rodrigues((hand_pose + mano.layer[hand_type].hand_mean).view(-1,3)).view(-1,3,3)
    pose = torch.cat([root,pose],dim=0).numpy()
    shape = shape.numpy().reshape(-1)
    return joint_img, joint_cam, mesh_img, mesh_cam, pose, shape


def get_mano_data_intertemp(mano_param, cam_param, do_flip, img_shape, mano):
    pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
    hand_type = mano_param['hand_type']
    pose = torch.FloatTensor(pose).view(-1, 3);
    shape = torch.FloatTensor(shape).view(1, -1);  # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

    if do_flip:
        if hand_type == 'right':
            hand_type = 'left'
        else:
            hand_type = 'right'

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3)
        root_pose = pose[mano.orig_root_joint_idx, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        new_R = np.dot(R, root_pose)
        root_pose, _ = cv2.Rodrigues(new_R)
        pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)
        new_R = torch.from_numpy(new_R)


    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in mano.orig_flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
        trans[:, 0] *= -1  # multiply -1

    # get root joint coordinate
    hand_pose = torch.cat((pose[:mano.orig_root_joint_idx, :], pose[mano.orig_root_joint_idx + 1:, :])).view(1, -1)
    zero_pose, _ = cv2.Rodrigues(np.eye(3))
    zero_pose = torch.from_numpy(zero_pose).view(1,3).float()
    zero_trans = torch.zeros(1,3)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=zero_pose, transl=zero_trans)
    mesh_coord = output.vertices[0].numpy()

    joint_coord = np.dot(mano.sh_joint_regressor, mesh_coord)

    # bring geometry to the original (before flip) position
    if do_flip:
        flip_trans_x = joint_coord[mano.sh_root_joint_idx, 0] * -2
        mesh_coord[:, 0] += flip_trans_x
        joint_coord[:, 0] += flip_trans_x

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)

    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                  dtype=np.float32).reshape(1, 3)
        R = torch.from_numpy(R)
        t = torch.from_numpy(t)

        t = t + torch.matmul(R, trans.transpose(0,1)).squeeze(-1)

        root_coord = joint_coord[mano.sh_root_joint_idx, None, :].copy()
        root_coord = torch.from_numpy(root_coord)
        mesh_coord = torch.from_numpy(mesh_coord) - root_coord
        joint_coord = torch.from_numpy(joint_coord) - root_coord

        mesh_coord = torch.matmul(mesh_coord,new_R.transpose(0,1)) + t.unsqueeze(0)
        mesh_coord = mesh_coord[0].numpy()
        joint_coord = torch.matmul(joint_coord, new_R.transpose(0, 1)) + t.unsqueeze(0)
        joint_coord = joint_coord[0].numpy()
    # flip translation
    if do_flip:  # avg of old and new root joint should be image center.
        focal, princpt = cam_param['focal'], cam_param['princpt']
        flip_trans_x = 2 * (
                ((img_shape[1] - 1) / 2. - princpt[0]) / focal[0] * joint_coord[mano.sh_root_joint_idx, 2]) - 2 * \
                       joint_coord[mano.sh_root_joint_idx][0]
        mesh_coord[:, 0] += flip_trans_x
        joint_coord[:, 0] += flip_trans_x
    # image projection
    mesh_cam = mesh_coord  # camera-centered 3D coordinates (not root-relative)
    joint_cam = joint_coord  # camera-centered 3D coordinates (not root-relative)

    mesh_img = cam2pixel(mesh_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = joint_img[:, :2]
    mesh_img = mesh_img[:, :2]
    if 'R' in cam_param:
        root = pose[mano.orig_root_joint_idx].view(1,3)
    else:
        root = zero_pose
    root = root.view(3)
    pose = hand_pose.view(45)
    pose = torch.cat([root,pose],dim=0)
    shape = shape.reshape(-1)
    return joint_img, joint_cam, mesh_img, mesh_cam, pose, shape


def get_mano_data_batch(mano_param, cam_param, mano):

    pose, shape, trans = mano_param['pose'].clone(), mano_param['shape'].clone(), mano_param['trans'].clone()
    bs = pose.shape[0]
    hand_type = mano_param['hand_type']
    pose = torch.FloatTensor(pose).view(bs, 16, 3)
    shape = torch.FloatTensor(shape).view(bs, 10) # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(bs, 3)  # translation vector

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    if 'R' in cam_param:
        R = torch.FloatTensor(cam_param['R']).reshape(1, 3, 3)
        root_pose = pose[:, mano.orig_root_joint_idx, :]
        root_pose = axis_to_Rmat(root_pose)
        root_pose = Rmat_to_axis(R @ root_pose)
        pose[:, mano.orig_root_joint_idx] = root_pose.view(bs, 3)

    # get root joint coordinate
    root_pose = pose[:, mano.orig_root_joint_idx].view(bs, 3)
    hand_pose = torch.cat((pose[:, :mano.orig_root_joint_idx, :], pose[:, mano.orig_root_joint_idx + 1:, :]),dim=1).view(bs, -1)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
    mesh_coord = output.vertices #bs x 778 x 3
    joint_regressor = torch.from_numpy(mano.sh_joint_regressor).unsqueeze(0)
    joint_coord = joint_regressor @ mesh_coord


    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    if 'R' in cam_param and 't' in cam_param:
        R, t = torch.FloatTensor(cam_param['R']).reshape(3, 3), torch.FloatTensor(cam_param['t']).reshape(1, 1, 3)

        root_coord = joint_coord[:, mano.sh_root_joint_idx, None, :].clone()
        joint_coord = joint_coord - root_coord + (R @ root_coord.transpose(-1, -2)).transpose(-1, -2) + t
        mesh_coord = mesh_coord - root_coord + (R @ root_coord.transpose(-1, -2)).transpose(-1, -2) + t


    mesh_cam = mesh_coord  # camera-centered 3D coordinates (not root-relative)
    joint_cam = joint_coord  # camera-centered 3D coordinates (not root-relative)
    mesh_img = cam2pixel_batch(mesh_cam,cam_param['focal'], cam_param['princpt'])
    joint_img = cam2pixel_batch(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = joint_img[..., :2]
    mesh_img = mesh_img[..., :2]
    return joint_img, joint_cam, mesh_img, mesh_cam


def get_mano_data_batch_V2(pose,shape,trans,cam_R,cam_T,focal,princpt,hand_type, mano):

    pose, shape, trans = pose.clone(), shape.clone(), trans.clone()
    bs = pose.shape[0]
    hand_type = hand_type
    pose = pose.view(bs, 16, 3)
    shape = shape.view(bs, 10) # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = trans.view(bs, 3)  # translation vector

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation

    R = cam_R.reshape(bs, 3, 3)
    root_pose = pose[:, mano.orig_root_joint_idx, :]
    root_pose = axis_to_Rmat(root_pose)
    root_pose = Rmat_to_axis(R @ root_pose)
    pose[:, mano.orig_root_joint_idx] = root_pose.view(bs, 3)


    # get root joint coordinate
    root_pose = pose[:, mano.orig_root_joint_idx].view(bs, 3)
    hand_pose = torch.cat((pose[:, :mano.orig_root_joint_idx, :], pose[:, mano.orig_root_joint_idx + 1:, :]),dim=1).view(bs, -1)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
    mesh_coord = output.vertices #bs x 778 x 3
    joint_regressor = torch.from_numpy(mano.sh_joint_regressor).unsqueeze(0).to(mesh_coord)
    joint_coord = joint_regressor @ mesh_coord


    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    R, t = cam_R.reshape(bs, 3, 3), cam_T.reshape(bs, 1, 3)

    root_coord = joint_coord[:, mano.sh_root_joint_idx, None, :].clone()
    joint_coord = joint_coord - root_coord + (R @ root_coord.transpose(-1, -2)).transpose(-1, -2) + t
    mesh_coord = mesh_coord - root_coord + (R @ root_coord.transpose(-1, -2)).transpose(-1, -2) + t


    mesh_cam = mesh_coord  # camera-centered 3D coordinates (not root-relative)
    joint_cam = joint_coord  # camera-centered 3D coordinates (not root-relative)
    mesh_img = cam2pixel_batch(mesh_cam,focal, princpt)
    joint_img = cam2pixel_batch(joint_cam, focal, princpt)
    joint_img = joint_img[..., :2]
    mesh_img = mesh_img[..., :2]
    return joint_img, joint_cam, mesh_img, mesh_cam


def axis_to_Rmat(axis, eps=1e-12):
    # axis: ... x 3
    # Rmat: ... x 3 x 3
    angle = torch.linalg.norm(axis, dim=-1, keepdim=True)  # ... x 1
    axes = axis / angle.clamp_min(eps)  # ... x 1
    sin = torch.sin(angle).unsqueeze(-1)  # ... x 1 x 1
    cos = torch.cos(angle).unsqueeze(-1)  # ... x 1 x 1
    x = axes[..., 0]
    y = axes[..., 1]
    z = axes[..., 2]
    o = torch.zeros_like(x)
    L = torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(list(x.shape) + [3, 3])  # ... x 3 x 3
    R = torch.eye(3).to(axis) + sin * L + (1 - cos) * (torch.matmul(axes.unsqueeze(-1), axes.unsqueeze(-2)) - torch.eye(3).to(axis))
    return R


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