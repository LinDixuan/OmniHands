import pickle
import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F

from .rotation import axis_to_Rmat, Rmat_to_axis, SE3_to_dualquat, dualquat_to_SE3, normalize_dualquat, \
    normalize_dualquat_to_SE3

from typing import Optional

PALM_BONE_LENGTH = 0.0947  # PALM_BONE_LENGTH when shape == 0

WRIST_IDX = [117, 118, 122, 38, 92, 234, 239, 279, 215, 214, 121, 78, 79, 108, 120, 119]

MANO_PARENT = [-1, 0, 1, 2, 3,
               0, 5, 6, 7,
               0, 9, 10, 11,
               0, 13, 14, 15,
               0, 17, 18, 19]

MANO_JOINT_COLOR = [[100, 100, 100],
                    [100, 0, 0],
                    [150, 0, 0],
                    [200, 0, 0],
                    [255, 0, 0],
                    [100, 100, 0],
                    [150, 150, 0],
                    [200, 200, 0],
                    [255, 255, 0],
                    [0, 100, 50],
                    [0, 150, 75],
                    [0, 200, 100],
                    [0, 255, 125],
                    [0, 50, 100],
                    [0, 75, 150],
                    [0, 100, 200],
                    [0, 125, 255],
                    [100, 0, 100],
                    [150, 0, 150],
                    [200, 0, 200],
                    [255, 0, 255]]


def fix_shape(left_mano, right_mano):
    if torch.sum(torch.abs(left_mano.shapedirs[:, 0, :] - right_mano.shapedirs[:, 0, :])) < 1:
        # before: left_mano.shapedirs[:, 0, :] ==  right_mano.shapedirs[:, 0, :]
        # after:  left_mano.shapedirs[:, 0, :] == -right_mano.shapedirs[:, 0, :]
        #print('Fix shapedirs bug of MANO')
        left_mano.shapedirs[:, 0, :] *= -1
    return left_mano, right_mano


def convert_mano_pkl(loadPath, savePath):
    # in original MANO pkl file, 'shapedirs' component is a chumpy object, convert it to a numpy array
    manoData = pickle.load(open(loadPath, 'rb'), encoding='latin1')
    output = {}
    manoData['shapedirs'].r
    for (k, v) in manoData.items():
        if k == 'shapedirs':
            output['shapedirs'] = v.r
        else:
            output[k] = v
    pickle.dump(output, open(savePath, 'wb'))


# def get_trans(old_z, new_z):
#     # z: bs x 3
#     x = torch.cross(old_z, new_z)
#     x = x / torch.norm(x, dim=1, keepdim=True).clamp_min(1e-8)
#     old_y = torch.cross(old_z, x)
#     new_y = torch.cross(new_z, x)
#     old_frame = torch.stack((x, old_y, old_z), dim=2)
#     new_frame = torch.stack((x, new_y, new_z), dim=2)
#     trans = torch.matmul(new_frame, old_frame.permute(0, 2, 1))
#     return trans


# def build_mano_frame(skelBatch):
#     # skelBatch: bs x 21 x 3
#     bs = skelBatch.shape[0]
#     mano_son = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]  # 15
#     mano_parent = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]  # 15
#     mano_order = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]  # 21

#     skel = skelBatch[:, mano_order]
#     twist = skel[:, mano_son] - skel[:, 1:16]  # bs x 15 x 3
#     twist = F.normalize(twist, dim=-1)

#     y = torch.zeros_like(twist)
#     y[..., 1] = 1.0
#     y[:, [12, 13, 14], 0] = 0.0
#     y[:, [12, 13, 14], 1] = np.sin(15 / 180 * np.pi)
#     y[:, [12, 13, 14], 2] = np.cos(15 / 180 * np.pi)
#     bend = torch.cross(y, twist, dim=-1)
#     # bend = torch.cross(twist, skel[:, mano_parent] - skel[:, 1:16], dim=-1)
#     bend = F.normalize(bend, dim=-1)

#     splay = torch.cross(twist, bend, dim=-1)
#     frame = torch.stack([bend, splay, twist], dim=-1)
#     return frame


class ManoLayer(Module):
    def __init__(self, manoPath: str, center_idx: Optional[int] = 0) -> None:
        super(ManoLayer, self).__init__()

        self.center_idx = center_idx

        manoData = pickle.load(open(manoPath, 'rb'), encoding='latin1')

        self.new_order = [0,
                          13, 14, 15, 16,
                          1, 2, 3, 17,
                          4, 5, 6, 18,
                          10, 11, 12, 19,
                          7, 8, 9, 20]

        # 45 * 45: PCA mat
        self.register_buffer('hands_components', torch.from_numpy(manoData['hands_components'].astype(np.float32)))
        hands_components_inv = torch.inverse(self.hands_components)
        self.register_buffer('hands_components_inv', hands_components_inv)
        # 16 * 778, J_regressor is a scipy csc matrix
        J_regressor = manoData['J_regressor'].tocoo(copy=False)
        location = []
        data = []
        for i in range(J_regressor.data.shape[0]):
            location.append([J_regressor.row[i], J_regressor.col[i]])
            data.append(J_regressor.data[i])
        i = torch.LongTensor(location)
        v = torch.FloatTensor(data)
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i.t(), v, torch.Size([16, 778])).to_dense(),
                             persistent=False)
        # 16 * 3
        self.register_buffer('J_zero', torch.from_numpy(manoData['J'].astype(np.float32)), persistent=False)
        # 778 * 16
        self.register_buffer('weights', torch.from_numpy(manoData['weights'].astype(np.float32)), persistent=False)
        # (778, 3, 135)
        self.register_buffer('posedirs', torch.from_numpy(manoData['posedirs'].astype(np.float32)), persistent=False)
        # (778, 3)
        self.register_buffer('v_template', torch.from_numpy(manoData['v_template'].astype(np.float32)),
                             persistent=False)
        # (778, 3, 10) shapedirs is <class 'chumpy.reordering.Select'>
        if isinstance(manoData['shapedirs'], np.ndarray):
            self.register_buffer('shapedirs', torch.Tensor(manoData['shapedirs']).float(), persistent=False)
        else:
            self.register_buffer('shapedirs', torch.Tensor(manoData['shapedirs'].r.copy()).float(), persistent=False)
        # 45
        self.register_buffer('hands_mean', torch.from_numpy(manoData['hands_mean'].astype(np.float32)),
                             persistent=False)

        self.faces = manoData['f'].astype(np.int64)  # 1538 * 3: faces

        self.finger_tips_idx = [745, 333, 444, 555, 672]
        # self.finger_tips_idx = [745, 317, 444, 556, 673]  # original manopth idx

        if self.v_template[270, 0] - self.v_template[216, 0] < 0:
            self.side = 'right'
        else:
            self.side = 'left'

        self.parent = [-1, ]
        for i in range(1, 16):
            self.parent.append(manoData['kintree_table'][0, i])

    def get_joint_regressor(self):
        '''
        output: 21 x 778
        '''
        J_r = self.J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_r[:5])
        for i in range(5):
            tip_regressor[i, self.finger_tips_idx[i]] = 1.0
        J_r = torch.cat([J_r, tip_regressor], dim=0)
        J_r = J_r[self.new_order].contiguous()
        return J_r

    def get_faces(self, close_wrist=False):
        if not close_wrist:
            return self.faces
        else:
            wrist = []
            for i in range(1, len(WRIST_IDX) - 1):
                wrist.append([WRIST_IDX[0], WRIST_IDX[i], WRIST_IDX[i + 1]])
            wrist = np.array(wrist)
            if self.side == 'right':
                wrist = wrist[:, [0, 2, 1]]
            return np.concatenate([self.faces, wrist], axis=0)

    def get_mean_pose_pca(self):
        return torch.zeros((1, 45)).to(self.hands_mean)

    def get_flat_pose_pca(self):
        return self.Rmat2pca(torch.eye((3)).repeat(1, 15, 1, 1).to(self.hands_mean))

    def pca2axis(self, pca):
        rotation_axis = pca.mm(self.hands_components[:pca.shape[1]])  # bs * 45
        rotation_axis = rotation_axis + self.hands_mean
        return rotation_axis  # bs * 45

    def pca2Rmat(self, pca):
        return self.axis2Rmat(self.pca2axis(pca))

    def axis2Rmat(self, axis):
        # axis: bs x 45
        return axis_to_Rmat(axis.reshape(-1, 15, 3))

    def axis2pca(self, axis):
        # axis: bs x 45
        pca = axis - self.hands_mean
        pca = pca.mm(self.hands_components_inv)
        return pca

    def Rmat2pca(self, R):
        # R: bs x 15 x 3 x 3
        return self.axis2pca(self.Rmat2axis(R))

    def Rmat2axis(self, R):
        # R: bs x 3 x 3
        return Rmat_to_axis(R).reshape(-1, 45)

    @torch.no_grad()
    def get_local_frame(self, shape):
        _, joints = self.forward(shape=shape)
        # skelBatch: bs x 21 x 3
        mano_son = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]  # 15
        mano_parent = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]  # 15
        mano_order = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]  # 21
        skel = joints[:, mano_order]
        twist = skel[:, mano_son] - skel[:, 1:16]  # bs x 15 x 3
        twist = F.normalize(twist, dim=-1)
        if self.side == 'left':
            twist = -twist
        up = torch.zeros_like(twist)
        up[..., 1] = 1.0
        up[:, [12, 13, 14], 0] = 0.0  # for thumb
        up[:, [12, 13, 14], 1] = np.sin(15 / 180 * np.pi)
        up[:, [12, 13, 14], 2] = np.cos(15 / 180 * np.pi)
        if self.side == 'left':
            up[..., 1] = -up[..., 1]
            up[..., 2] = -up[..., 2]
        bend = torch.cross(up, twist, dim=-1)
        bend = F.normalize(bend, dim=-1)
        splay = torch.cross(twist, bend, dim=-1)
        return bend, splay, twist  # bs x 15 x 3

    @staticmethod
    def buildSE3_batch(R, t):
        # R: bs * 3 * 3
        # t: bs * 3 * 1
        # return: bs * 4 * 4
        bs = R.shape[0]
        pad = torch.zeros((bs, 1, 4), dtype=R.dtype, device=R.device)
        pad[:, 0, 3] = 1.0
        temp = torch.cat([R, t], 2)  # bs * 3 * 4
        return torch.cat([temp, pad], 1)

    @staticmethod
    def SE3_apply(SE3, v):
        # SE3: bs * 4 * 4
        # v: bs * 3
        # return: bs * 3
        bs = v.shape[0]
        pad = torch.ones((bs, 1), dtype=v.dtype, device=v.device)
        temp = torch.cat([v, pad], 1).unsqueeze(2)  # bs * 4 * 1
        return SE3.bmm(temp)[:, :3, 0]

    def forward(self,
                root_rotation: Optional[torch.Tensor] = None,
                pose: Optional[torch.Tensor] = None,
                shape: Optional[torch.Tensor] = None,
                trans: Optional[torch.Tensor] = None,
                use_pca: bool = True,
                need_vertex_deform: bool = False,
                dual_quat_skin: bool = False,
                center_idx: Optional[int] = -1) -> tuple:
        '''
        input
            root_rotation : bs x 3 x 3 or bs x 3
            pose : bs * ncomps or bs x 15 x 3 x 3 or bs x 45
            shape : bs x 10
            trans : bs x 3
        '''
        bs = None
        for x in [root_rotation, pose, shape, trans]:
            if x is not None:
                bs = x.shape[0]
                break
        if bs is None:
            bs = 1
        if root_rotation is None:
            root_rotation = torch.eye((3)).repeat(bs, 1, 1).to(self.hands_mean)
        else:
            if root_rotation.dim() == 2 and root_rotation.shape[-1] == 3:
                root_rotation = axis_to_Rmat(root_rotation)

        if pose is None:
            rotation_mat = torch.eye((3)).repeat(bs, 15, 1, 1).to(self.hands_mean)
        else:
            if tuple(pose.shape[1:]) == (15, 3, 3):
                rotation_mat = pose
            else:
                if use_pca or pose.shape[-1] < 45:
                    rotation_mat = self.pca2Rmat(pose)
                else:
                    rotation_mat = self.axis2Rmat(pose)

        if shape is None:
            shape = torch.zeros(bs, 10).to(self.hands_mean)

        shapeBlendShape = torch.matmul(self.shapedirs, shape.permute(1, 0)).permute(2, 0, 1)
        v_shaped = self.v_template + shapeBlendShape  # bs * 778 * 3
        j_tpose = torch.matmul(self.J_regressor, v_shaped)  # bs * 16 * 3

        Imat = torch.eye(3, dtype=rotation_mat.dtype, device=rotation_mat.device).repeat(bs, 15, 1, 1)
        pose_shape = rotation_mat.reshape(bs, -1) - Imat.reshape(bs, -1)  # bs * 135
        poseBlendShape = torch.matmul(self.posedirs, pose_shape.permute(1, 0)).permute(2, 0, 1)
        v_tpose = v_shaped + poseBlendShape  # bs * 778 * 3

        SE3_j = []
        R = root_rotation
        t = (torch.eye(3, dtype=rotation_mat.dtype, device=rotation_mat.device).repeat(bs, 1, 1) - R).bmm(
            j_tpose[:, 0].unsqueeze(2))
        SE3_j.append(self.buildSE3_batch(R, t))
        for i in range(1, 16):
            R = rotation_mat[:, i - 1]
            t = (torch.eye(3, dtype=rotation_mat.dtype, device=rotation_mat.device).repeat(bs, 1, 1) - R).bmm(
                j_tpose[:, i].unsqueeze(2))
            SE3_local = self.buildSE3_batch(R, t)
            SE3_j.append(torch.matmul(SE3_j[self.parent[i]], SE3_local))
        SE3_j = torch.stack(SE3_j, dim=1)  # bs * 16 * 4 * 4

        j_withoutTips = []
        j_withoutTips.append(j_tpose[:, 0])
        for i in range(1, 16):
            j_withoutTips.append(self.SE3_apply(SE3_j[:, self.parent[i]], j_tpose[:, i]))

        if dual_quat_skin:
            dual_quat_j = SE3_to_dualquat(SE3_j)
            for i in range(1, 16):
                need_reverse = torch.sum(dual_quat_j[:, i, :4] * dual_quat_j[:, self.parent[i], :4], dim=-1) < 0
                dual_quat_j[need_reverse, i] = -dual_quat_j[need_reverse, i]
            dual_quat_v = torch.matmul(self.weights, dual_quat_j)
            SE3_v = normalize_dualquat_to_SE3(dual_quat_v)
        else:
            SE3_v = torch.matmul(self.weights, SE3_j.view(bs, 16, 16)).view(bs, -1, 4, 4)  # bs * 778 * 4 * 4
            # print('SE3_v', SE3_v[0, 606])

        if need_vertex_deform:
            temp = (poseBlendShape + shapeBlendShape).unsqueeze(-1)  # bs x 778 x 3 x 1
            deformR = SE3_v[:, :, :3, :3]  # bs x 778 x 3 x 3
            deformT = SE3_v[:, :, :3, 3:4] + torch.matmul(SE3_v[:, :, :3, :3], temp)  # bs x 778 x 3 x 1
            deformT = deformT[..., 0]

        v_output = SE3_v[:, :, :3, :3].matmul(v_tpose.unsqueeze(3)) + SE3_v[:, :, :3, 3:4]
        v_output = v_output[:, :, :, 0]  # bs * 778 * 3

        jList = j_withoutTips
        jList = jList + [v_output[:, self.finger_tips_idx[i]] for i in range(5)]
        j_output = torch.stack(jList, dim=1)
        j_output = j_output[:, self.new_order]

        if center_idx is not None and center_idx < 0:
            center_idx = self.center_idx
        if center_idx is not None:
            center = j_output[:, center_idx:(center_idx + 1)]
            v_output = v_output - center
            j_output = j_output - center
            if need_vertex_deform:
                deformT = deformT - center

        if trans is not None:
            trans = trans.unsqueeze(1)  # bs * 1 * 3
            v_output = v_output + trans
            j_output = j_output + trans
            if need_vertex_deform:
                deformT = deformT + trans

        if need_vertex_deform:
            return v_output, j_output, deformR, deformT.unsqueeze(-1)
        else:
            return v_output, j_output
