import math
import torch
import torch.nn.functional as F

from typing import Optional

__all__ = ['axis_to_quat', 'quat_to_axis',
           'quat_to_Rmat', 'Rmat_to_quat',
           'Rmat_to_axis', 'axis_to_Rmat',
           'Rmat_to_vec6', 'vec6_to_Rmat'

                           'normalize_rotation',
           'calc_rotaion_norm_loss',

           'SE3_to_dualquat', 'dualquat_to_SE3',
           'apply_dualquat', 'normalize_dualquat', 'normalize_dualquat_to_SE3',
           'Slerp_SE3'
           ]


def __sin_divide_angle(angle: torch.Tensor, sin=None, eps: float = 1e-3) -> torch.Tensor:
    small_angle = angle < eps
    large_angle = ~small_angle
    x = torch.empty_like(angle)
    x[small_angle] = 1 - angle[small_angle] ** 2 / 6 + angle[small_angle] ** 4 / 120
    if sin is not None:
        x[large_angle] = sin[large_angle] / angle[large_angle]
    else:
        x[large_angle] = torch.sin(angle[large_angle]) / angle[large_angle]
    return x


def __angle_divide_sin(angle: torch.Tensor, sin=None, eps: float = 1e-3) -> torch.Tensor:
    small_angle = angle < eps
    large_angle = ~small_angle
    x = torch.empty_like(angle)
    x[small_angle] = 1 + angle[small_angle] ** 2 / 6 + angle[small_angle] ** 4 * 7 / 360
    if sin is not None:
        x[large_angle] = angle[large_angle] / sin[large_angle]
    else:
        x[large_angle] = angle[large_angle] / torch.sin(angle[large_angle])
    return x


def axis_to_quat(axis: torch.Tensor) -> torch.Tensor:
    # axis: ... x 3
    angle_2 = torch.linalg.norm(axis, dim=-1, keepdim=True) * 0.5  # ... x 1
    scale = 0.5 * __sin_divide_angle(angle_2)
    quat = torch.cat([torch.cos(angle_2), axis * scale], dim=-1)
    quat = F.normalize(quat, dim=-1)
    return quat


def quat_to_axis(quat: torch.Tensor) -> torch.Tensor:
    quat = quat * torch.sign(quat[..., :1])  # make sure w > 0, which means 0 < angle < pi
    angle = torch.arctan2(torch.linalg.norm(quat[..., 1:], dim=-1, keepdim=True), quat[..., 0:1])
    scale = 2 * __angle_divide_sin(angle)
    axis = quat[..., 1:] * scale
    return axis


def quat_to_Rmat(quat: torch.Tensor) -> torch.Tensor:
    cos_2 = quat[..., :1].unsqueeze(-1)  # ... x 1 x 1
    sin_2_x_L = quat[..., 1:]  # ... x 3
    x = sin_2_x_L[..., 0]
    y = sin_2_x_L[..., 1]
    z = sin_2_x_L[..., 2]
    o = torch.zeros_like(x)
    L = torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(list(x.shape) + [3, 3])  # ... x 3 x 3
    R = torch.eye(3).to(quat) + 2 * cos_2 * L + 2 * torch.matmul(L, L)
    return R


def Rmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    shape = list(R.shape[:-2])
    R = R.reshape([-1, 3, 3])

    decision_matrix = torch.empty([R.shape[0], 4]).to(R)

    decision_matrix[..., :3] = torch.diagonal(R, dim1=-2, dim2=-1)
    decision_matrix[..., -1] = torch.sum(decision_matrix[..., :3], dim=-1)
    choices = decision_matrix.argmax(axis=-1)

    quat = torch.empty_like(decision_matrix)

    ind = torch.nonzero(choices != 3).reshape(-1)
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3
    quat[ind, 0] = R[ind, k, j] - R[ind, j, k]
    quat[ind, i + 1] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j + 1] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k + 1] = R[ind, k, i] + R[ind, i, k]

    ind = torch.nonzero(choices == 3).reshape(-1)
    quat[ind, 0] = 1 + decision_matrix[ind, -1]
    quat[ind, 1] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 2] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 3] = R[ind, 1, 0] - R[ind, 0, 1]

    quat = F.normalize(quat, dim=-1)
    quat = quat.reshape(shape + [4])
    return quat


def Rmat_to_axis(R: torch.Tensor) -> torch.Tensor:
    return quat_to_axis(Rmat_to_quat(R))


def axis_to_Rmat(R: torch.Tensor) -> torch.Tensor:
    return quat_to_Rmat(axis_to_quat(R))


def vec6_to_Rmat(vec6: torch.Tensor) -> torch.Tensor:
    # # vec6: ... x 6
    # # Rmat: ... x 3 x 3
    # x = vec6[..., 0:3]
    # y = vec6[..., 3:6]
    # x = F.normalize(x, dim=-1)
    # y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    # y = F.normalize(y, dim=-1)
    # z = torch.cross(x, y, dim=-1)

    # vec6: ... x 6
    # Rmat: ... x 3 x 3
    x = vec6[..., 0:3]
    y = vec6[..., 3:6]
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    middle = F.normalize(x + y, dim=-1)
    orthmid = F.normalize(x - y, dim=-1)
    s = math.sqrt(2) * 0.5
    x = s * (middle + orthmid)
    y = s * (middle - orthmid)
    z = torch.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)


def Rmat_to_vec6(Rmat: torch.Tensor) -> torch.Tensor:
    # vec6: ... x 6
    # Rmat: ... x 3 x 3
    return torch.cat((Rmat[..., 0], Rmat[..., 1]), dim=-1)


def unbiased_GS(R: torch.Tensor) -> torch.Tensor:
    # R: ... x 3 x 3
    assert R.shape[-2:] == (3, 3)
    t1 = R[..., 0]
    t2 = R[..., 1]
    t3 = R[..., 2]

    r1 = F.normalize((torch.cross(t2, t3, dim=-1) + t1) / 2, dim=-1)
    r2 = (torch.cross(t3, r1, dim=-1) + t2) / 2
    r2 = F.normalize(r2 - torch.sum(r1 * r2, dim=-1, keepdim=True) * r1, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    Rmat = torch.stack([r1, r2, r3], dim=-1)
    return Rmat


def normalize_rotation(x: torch.Tensor, return_R: bool = False,
                       is_axis: bool = False) -> torch.Tensor:
    if x.shape[-1] == 4:  # quaternions
        x = F.normalize(x, dim=-1)
        if return_R:
            return quat_to_Rmat(x)
        else:
            return x
    elif x.shape[-1] == 6:  # 6-dim rotation vector
        R = vec6_to_Rmat(x)
        if return_R:
            return R
        else:
            return Rmat_to_vec6(R)
    elif x.shape[-1] == 3:
        if x.shape[-2] != 3 or is_axis:
            R = axis_to_Rmat(x)  # 3-dim rotation vector
            if return_R:
                return R
            else:
                return Rmat_to_axis(R)
        else:
            R = unbiased_GS(x)  # rotation matrix
            return R
    else:
        raise NotImplementedError


def calc_rotaion_norm_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 4:  # quaternions
        loss = ((torch.linalg.norm(x, dim=-1) - 1) ** 2).mean()  # |x| == 1
        return loss
    elif x.shape[-1] == 6:  # 6-dim rotation vector
        x, y = x[..., :3], x[..., 3:]
        loss = 0
        loss = loss + ((torch.linalg.norm(x, dim=-1) - 1) ** 2).mean()  # |x| == 1
        loss = loss + ((torch.linalg.norm(y, dim=-1) - 1) ** 2).mean()  # |y| == 1
        loss = loss + 2 * (torch.sum(x * y, dim=-1) ** 2).mean()  # <x, y> == 0
        return loss
    elif x.shape[-2:] == (3, 3):  # rotation matrix
        loss = 0
        loss = loss + 2 * ((torch.matmul(x, x.transpose(-1, -2)) - torch.eye(3).to(x)) ** 2).mean()  # RR^T == I
        x, y, z = x[..., 0], x[..., 1], x[..., 2]
        det = torch.sum(torch.cross(x, y, dim=-1) * z, dim=-1)
        loss = loss + ((det - 1) ** 2).mean()  # |R| == 1
        return loss
    else:
        raise NotImplementedError


def quat_prod_quat(q1, q2):
    a = q1[..., :1] * q2[..., :1] - torch.sum(q1[..., 1:] * q2[..., 1:], dim=-1, keepdim=True)
    b = q1[..., :1] * q2[..., 1:] + q2[..., :1] * q1[..., 1:] + torch.cross(q1[..., 1:], q2[..., 1:], dim=-1)
    return torch.cat([a, b], dim=-1)


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def dualquat_prod_dualquat(dq1, dq2):
    dq1_real = dq1[..., :4]
    dq1_com = dq1[..., 4:]
    dq2_real = dq2[..., :4]
    dq2_com = dq2[..., 4:]
    real = quat_prod_quat(dq1_real, dq2_real)
    com = quat_prod_quat(dq1_com, dq2_real) + quat_prod_quat(dq1_real, dq2_com)
    return torch.cat([real, com], dim=-1)


def dualquat_conj(dq):
    return torch.cat([quat_conj(dq[..., :4]), -quat_conj(dq[..., 4:])], dim=-1)


def points_to_dualquat(x):
    a = torch.zeros_like(x[..., :3])
    b = torch.zeros_like(x[..., :2])
    out = torch.cat([a, b, x[..., :3]], dim=-1)
    out[..., 0] = 1.0
    return out


def apply_dualquat(dq: torch.Tensor, x: torch.Tensor) -> torch.tensor:
    x_dq = points_to_dualquat(x)
    x_dq = dualquat_prod_dualquat(dualquat_prod_dualquat(dq, x_dq), dualquat_conj(dq))
    return x_dq[..., -3:]


def SE3_to_dualquat(SE3: torch.Tensor) -> torch.tensor:
    R = SE3[..., :3, :3]
    T = SE3[..., :3, 3]
    R_quat = Rmat_to_quat(R)

    T_quat = torch.cat([torch.zeros_like(T[..., :1]), 0.5 * T], dim=-1)
    return torch.cat([R_quat, quat_prod_quat(T_quat, R_quat)], dim=-1)


def dualquat_to_SE3(dualquat: torch.Tensor) -> torch.tensor:
    real_quat = dualquat[..., :4]
    com_quat = dualquat[..., 4:]
    real_quat_inv = quat_conj(real_quat)
    R = quat_to_Rmat(real_quat)
    T = 2 * quat_prod_quat(com_quat, real_quat_inv)[..., 1:]
    mat = torch.cat([R, T.unsqueeze(-1)], dim=-1)
    SE3 = torch.cat([mat, torch.zeros_like(mat[..., :1, :])], dim=-2)
    SE3[..., 3, 3] = 1.0
    return SE3


def normalize_dualquat(dq: torch.Tensor, eps: float = 1e-12) -> torch.tensor:
    real_quat = dq[..., :4]  # ... x 4
    com_quat = dq[..., 4:]  # ... x 4
    a = torch.linalg.norm(real_quat, dim=-1, keepdim=True)  # ... x 1
    b = torch.sum(real_quat * com_quat, dim=-1, keepdim=True)  # ... x 1

    real_output = F.normalize(real_quat, dim=-1)
    com_output = com_quat / a.clamp_min(eps) - real_quat * b / (a ** 3).clamp_min(eps)
    return torch.cat([real_output, com_output], dim=-1)


def normalize_dualquat_to_SE3(dq: torch.Tensor, eps: float = 1e-7) -> torch.tensor:
    real_quat = dq[..., :4]  # ... x 4
    com_quat = dq[..., 4:]  # ... x 4

    com_quat = com_quat / torch.linalg.norm(real_quat, dim=-1, keepdim=True).clamp_min(eps)
    real_quat = F.normalize(real_quat, dim=-1)

    R = quat_to_Rmat(real_quat)
    T = 2 * quat_prod_quat(com_quat, quat_conj(real_quat))[..., 1:]

    mat = torch.cat([R, T.unsqueeze(-1)], dim=-1)
    SE3 = torch.cat([mat, torch.zeros_like(mat[..., :1, :])], dim=-2)
    SE3[..., 3, 3] = 1.0
    return SE3


# TODO: check this method
class Slerp_SE3():
    def __init__(self, R: torch.Tensor, T: Optional[torch.Tensor] = None) -> None:
        '''
        R: [...] x T x 3 x 3
        T: [...] x T x 3 or None
        '''
        if T is None:
            T = torch.zeros_like(R.shape[:-1]).to(R)  # [...] x T x 3
        assert R.shape[:-2] == T.shape[:-1]
        assert T.shape[-1] == 3

        self.times = R.shape[-3]

        self.R_init = R  # [...] x T x 3 x 3
        self.T_init = T.unsqueeze(-1)  # [...] x T x 3 x 1

        delta_R = torch.matmul(self.R_init[..., 1:, :, :],
                               self.R_init[..., :-1, :, :].transpose(-1, -2))  # [...] x T-1 x 3 x 3
        delta_T = self.T_init[..., 1:, :, :] - torch.matmul(delta_R, self.T_init[..., :-1, :, :])  # [...] x T-1 x 3 x 1

        R_zero = torch.eye(3).expand_as(delta_R[..., :1, :, :])
        T_zero = torch.zeros_like(delta_T[..., :1, :, :])
        delta_R = torch.cat([delta_R, R_zero], dim=-3)  # [...] x T x 3 x 3
        delta_T = torch.cat([delta_T, T_zero], dim=-3)  # [...] x T x 3 x 1

        delta_SE3 = torch.cat([delta_R, delta_T], dim=-1)  # [...] x T x 3 x 4
        delta_dual_quat = SE3_to_dualquat(delta_SE3)  # [...] x T x 8
        self.delta_dual_quat = delta_dual_quat * delta_dual_quat[..., :1].sign()  # [...] x T x 8

    def __call__(self, times: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        input:
            times: [...] x sample_times
        output:
            R: [...] x sample_times x 3 x 3
            T: [...] x sample_times x 3 x 1
        '''
        assert times.shape[:-1] == self.R_init.shape[:-3]
        assert (times >= 0).all() and (times <= 1).all()

        times = times * (self.times - 1)

        times_floor = times.long()  # [...] x Ts
        alpha = times - times_floor  # [...] x Ts

        R_init = torch.gather(self.R_init, dim=-3, index=times_floor.unsqueeze(-1).unsqueeze(-1).expand(
            list(times_floor.shape) + [3, 3]))  # [...] x Ts x 3 x 3
        T_init = torch.gather(self.T_init, dim=-3, index=times_floor.unsqueeze(-1).unsqueeze(-1).expand(
            list(times_floor.shape) + [3, 1]))  # [...] x Ts x 3 x 1
        delta_dual_quat = torch.gather(self.delta_dual_quat, dim=-2, index=times_floor.unsqueeze(-1).expand(
            list(times_floor.shape) + [8]))  # [...] x Ts x 8

        zero_dual_quat = torch.zeros_like(delta_dual_quat)
        zero_dual_quat[..., 0] = 1.0

        alpha = alpha.unsqueeze(-1)

        delta_dual_quat = alpha * delta_dual_quat + (1 - alpha) * zero_dual_quat

        SE3 = normalize_dualquat_to_SE3(delta_dual_quat)

        delta_R = SE3[..., :3, :3]  # [...] x Ts x 3 x 3
        delta_T = SE3[..., :3, 3:4]  # [...] x Ts x 3 x 3

        R = torch.matmul(delta_R, R_init)
        T = torch.matmul(delta_R, T_init) + delta_T

        return R, T