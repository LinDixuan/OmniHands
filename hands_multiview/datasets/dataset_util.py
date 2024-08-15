import numpy as np
from PIL import Image
import cv2
import random
import torchvision
import torchvision.transforms.functional as F

def get_fw_weight(radius, motion_rand):
    w = np.arange(radius + 1)[::-1]
    w = 0.7 ** w * motion_rand
    w[-1] = 1.
    w /= w.sum()
    return w

def transform_coords(pts, affine_trans):
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows


def transform_img(img, affine_trans, res):
    trans = np.linalg.inv(affine_trans)
    img = img.transform(
        tuple(res),
        Image.AFFINE,
        (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]),
    )
    return img


def get_affine_transform(center, scale, res, rot=0, K=None):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    origin_rot_center = rot_mat.dot(
        center.tolist()
        + [
            1,
        ]
    )[:2]
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    if K is not None:
        t_mat = np.eye(3)
        t_mat[0, 2] = -K[0, 2]
        t_mat[1, 2] = -K[1, 2]
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        transformed_center = (
            t_inv.dot(rot_mat)
            .dot(t_mat)
            .dot(
                center.tolist()
                + [
                    1,
                ]
            )
        )
        affinetrans_post_rot = get_affine_trans_no_rot(
            transformed_center[:2], scale, res
        )
        return (
            total_trans.astype(np.float32),
            affinetrans_post_rot.astype(np.float32),
            rot_mat.astype(np.float32),
        )
    else:
        return total_trans.astype(np.float32), rot_mat.astype(np.float32)


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + 0.5)
    affinet[2, 2] = 1
    return affinet


def rotation_angle(angle, rot_mat, coord_change_mat=None):
    per_rdg, _ = cv2.Rodrigues(angle)
    if coord_change_mat is not None:
        rot_mat = np.dot(rot_mat, coord_change_mat)
    resrot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
    return resrot[:, 0].astype(np.float32)


def get_bbox_joints(joints2d, bbox_factor=1.0):
    min_x, min_y = joints2d.min(0)
    max_x, max_y = joints2d.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    bbox_delta_x = (max_x - min_x) * bbox_factor / 2
    bbox_delta_y = (max_y - min_y) * bbox_factor / 2
    bbox_delta = np.asarray([bbox_delta_x, bbox_delta_y])
    bbox = np.array([*(center - bbox_delta), *(center + bbox_delta)], dtype=np.float32)
    return bbox


def normalize_joints(joints2d, bbox):
    bbox = bbox.reshape(2, 2)
    joints2d = (joints2d - bbox[0, :]) / (bbox[1, :] - bbox[0, :])
    joints2d[np.isnan(joints2d) + np.isinf(joints2d)] = -100
    outside_mask = ((joints2d < 0) + (joints2d > 1)).any(axis=1)
    joints2d[outside_mask] = -100
    return joints2d


def denormalize_joints(normalized_joints2d, bbox):
    mask = (normalized_joints2d >= 0.0).all(axis=1)
    bbox = bbox.reshape(2, 2)
    joints2d = normalized_joints2d * (bbox[1, :] - bbox[0, :])
    joints2d += bbox[0, :]
    return joints2d, mask


def get_mask_ROI(mask, bbox):
    mask = mask.crop(bbox)
    return mask


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def get_color_jitter_funcs(brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: F.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: F.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(lambda img: F.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: F.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)
    return img_transforms


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: F.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: F.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(lambda img: F.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: F.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img


def get_bbox21_3d_from_dict(vertex):
    bbox21_3d = {}
    for key in vertex:
        vp = vertex[key][:]
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = (
            np.max(x),
            np.min(x),
            np.max(y),
            np.min(y),
            np.max(z),
            np.min(z),
        )
        p_blb = np.array([x_min, y_min, z_min])  # bottem, left, behind
        p_brb = np.array([x_max, y_min, z_min])  # bottem, right, behind
        p_blf = np.array([x_min, y_max, z_min])  # bottem, left, front
        p_brf = np.array([x_max, y_max, z_min])  # bottem, right, front
        #
        p_tlb = np.array([x_min, y_min, z_max])  # top, left, behind
        p_trb = np.array([x_max, y_min, z_max])  # top, right, behind
        p_tlf = np.array([x_min, y_max, z_max])  # top, left, front
        p_trf = np.array([x_max, y_max, z_max])  # top, right, front
        #
        p_center = (p_tlb + p_brf) / 2
        #
        p_ble = (p_blb + p_blf) / 2  # bottem, left, edge center
        p_bre = (p_brb + p_brf) / 2  # bottem, right, edge center
        p_bfe = (p_blf + p_brf) / 2  # bottem, front, edge center
        p_bbe = (p_blb + p_brb) / 2  # bottem, behind, edge center
        #
        p_tle = (p_tlb + p_tlf) / 2  # top, left, edge center
        p_tre = (p_trb + p_trf) / 2  # top, right, edge center
        p_tfe = (p_tlf + p_trf) / 2  # top, front, edge center
        p_tbe = (p_tlb + p_trb) / 2  # top, behind, edge center
        #
        p_lfe = (p_tlf + p_blf) / 2  # left, front, edge center
        p_lbe = (p_tlb + p_blb) / 2  # left, behind, edge center
        p_rfe = (p_trf + p_brf) / 2  # left, front, edge center
        p_rbe = (p_trb + p_brb) / 2  # left, behind, edge center

        pts = np.stack(
            (
                p_blb,
                p_brb,
                p_blf,
                p_brf,
                p_tlb,
                p_trb,
                p_tlf,
                p_trf,
                p_ble,
                p_bre,
                p_bfe,
                p_bbe,
                p_tle,
                p_tre,
                p_tfe,
                p_tbe,
                p_lfe,
                p_lbe,
                p_rfe,
                p_rbe,
                p_center,
            )
        )
        bbox21_3d[key] = pts
    return bbox21_3d


def get_diameter(vertex):
    diameters = {}
    for key in vertex:
        vp = vertex[key][:]
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = (
            np.max(x),
            np.min(x),
            np.max(y),
            np.min(y),
            np.max(z),
            np.min(z),
        )
        diameter_x = abs(x_max - x_min)
        diameter_y = abs(y_max - y_min)
        diameter_z = abs(z_max - z_min)
        diameters[key] = np.sqrt(diameter_x**2 + diameter_y**2 + diameter_z**2)
    return diameters


def fuse_bbox(bbox_1, bbox_2, img_shape, scale_factor=1.0):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    scale = max_delta * scale_factor
    return center, scale


def regularize_bbox(bbox, img_size):
    w, h = img_size
    bbox[::2] = np.clip(bbox[::2], 0, w)
    bbox[1::2] = np.clip(bbox[1::2], 0, h)
    return bbox


def inter_area(bbox1, bbox2):
    inter_tl = np.maximum(bbox1[:2], bbox2[:2])
    inter_br = np.minimum(bbox1[2:], bbox2[2:])
    inter_w, inter_h = inter_br - inter_tl
    inter_area = max(inter_w, 0) * max(inter_h, 0)
    return inter_area


def square_pad(img, return_bbox=False):
    w, h = img.size
    max_wh = np.max([w, h])
    p_left, p_top = (max_wh - w) // 2, (max_wh - h) // 2
    p_right, p_bottom = max_wh - (w + p_left), max_wh - (h + p_top)
    padding = (p_left, p_top, p_right, p_bottom)
    if return_bbox:
        return F.pad(img, padding, 0, "constant"), np.asarray(
            [p_left, p_top, max_wh - p_right, max_wh - p_bottom]
        )
    return F.pad(img, padding, 0, "constant")


def get_bbox_corners(bbox):
    x1, y1, x2, y2 = bbox
    bbox_corners = np.asanyarray(
        [
            [x1, y1],
            [x1, y2],
            [x2, y1],
            [x2, y2],
        ]
    )
    return bbox_corners
