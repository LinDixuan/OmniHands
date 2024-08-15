from typing import Optional, List
import os
import cv2 as cv
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.modules.utils import sample_feat_from_image

# from pytorch3d.utils import TensorProperties
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, OrthographicCameras,
    TexturesUV, TexturesVertex,

    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader, SoftPhongShader, SoftSilhouetteShader,
    MeshRenderer, MeshRendererWithFragments,

    AmbientLights,
    PointLights,
)
from pytorch3d.ops import interpolate_face_attributes

# HAND_COLOR_L = [204, 153, 0]
HAND_COLOR_L = [102, 102, 255]
HAND_COLOR_R = [255, 255, 255]


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy_data', torch.empty(0), persistent=False)
        self.amblights = AmbientLights()
        self.point_lights = PointLights(location=[[0, 0, -1.5]])

    def device(self):
        return self.dummy_data.device

    def build_rasterizer(self, img_size: int = 256, soft: bool = False) -> MeshRasterizer:
        if soft:
            sigma = 1e-4
            raster_settings = RasterizationSettings(
                image_size=img_size,
                blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
                faces_per_pixel=50,
            )
        else:
            raster_settings = RasterizationSettings(
                image_size=img_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
        rasterizer = MeshRasterizer(raster_settings=raster_settings)
        return rasterizer.to(self.device())

    def build_rgb_renderer(self, img_size: int = 256, soft: bool = False) -> MeshRendererWithFragments:
        rasterizer = self.build_rasterizer(img_size, soft)
        if soft:
            renderer = MeshRendererWithFragments(
                rasterizer=rasterizer,
                shader=SoftPhongShader()
            )
        else:
            renderer = MeshRendererWithFragments(
                rasterizer=rasterizer,
                shader=HardPhongShader()
            )
        return renderer.to(self.device())

    def build_mask_renderer(self, img_size: int = 256, soft: bool = False) -> MeshRendererWithFragments:
        rasterizer = self.build_rasterizer(img_size, soft)
        renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=SoftSilhouetteShader()
        )
        return renderer.to(self.device())

    def build_camera(self,
                     cameras: Optional[torch.Tensor] = None, img_size: Optional[int] = None,
                     scale: Optional[torch.Tensor] = None, trans2d: Optional[torch.Tensor] = None):
        device = self.device()
        if cameras is not None:
            # cameras: bs x 3 x 3
            fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / img_size
            pps = -cameras[:, :2, -1] * 2 / img_size + 1
            return PerspectiveCameras(focal_length=fs.to(device),
                                      principal_point=pps.to(device),
                                      in_ndc=True,
                                      device=device)

        if scale is not None and trans2d is not None:
            # V2d (0~1) = Scale * img_size + 0.5 + 0.5 * Trans2d
            bs = scale.shape[0]
            R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(bs, 1, 1).to(scale.dtype)
            T = torch.tensor([0, 0, 10]).repeat(bs, 1).to(scale.dtype)
            return OrthographicCameras(focal_length=2 * scale.to(device),
                                       principal_point=-trans2d.to(device),
                                       R=R.to(device),
                                       T=T.to(device),
                                       in_ndc=True,
                                       device=device)

    def build_texture(self, uv_verts: Optional[torch.Tensor] = None, uv_faces: Optional[torch.Tensor] = None,
                      texture: Optional[torch.Tensor] = None,
                      v_color: Optional[torch.Tensor] = None):
        device = self.device()
        if uv_verts is not None and uv_faces is not None and texture is not None:
            return TexturesUV(texture.to(device), uv_faces.to(device), uv_verts.to(device))
        if v_color is not None:
            return TexturesVertex(verts_features=v_color.to(device))

    def render_face_attr_python(self, fragments, attributes):
        '''
        attributes: bs x f x 3 x D
        return:
            pixel_vals: bs x H x W x D
            alpha: bs x H x W
        '''
        bs, f, _, D = attributes.shape
        attributes = attributes.reshape(bs * f, 3, D)  # bs*f x 3 x D

        pix_to_face = fragments.pix_to_face  # bs x H x W x K
        bary_coords = fragments.bary_coords  # bs x H x W x K x 3
        mask = pix_to_face == -1
        pix_to_face_clone = pix_to_face.clone()
        pix_to_face_clone[mask] = 0
        bs, H, W, K, _ = bary_coords.shape
        idx = pix_to_face_clone.view(bs * H * W * K, 1, 1).expand(bs * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(bs, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0]  # bs x H x W x D
        return pixel_vals

    def render_face_attr(self, fragments, attributes):
        '''
        attributes: bs x f x 3 x D
        return:
            pixel_vals: bs x H x W x D
            alpha: bs x H x W
        '''
        bs, f, _, D = attributes.shape
        attributes = attributes.reshape(bs * f, 3, D)  # bs*f x 3 x D
        pix_to_face = fragments.pix_to_face  # bs x H x W x K
        bary_coords = fragments.bary_coords  # bs x H x W x K x 3
        pixel_vals = interpolate_face_attributes(pix_to_face, bary_coords, attributes)
        pixel_vals = pixel_vals[:, :, :, 0]  # bs x H x W x D
        return pixel_vals


class ManoRenderer_twohand(Renderer):
    def __init__(self,
                 img_size: int,
                 right_mano_faces: np.ndarray,
                 right_mano_uv: Optional[np.ndarray] = None,
                 right_mano_uvfaces: Optional[np.ndarray] = None):
        super().__init__()
        self.img_size = img_size

        faces = right_mano_faces
        self.register_buffer('left_faces', torch.tensor(faces[..., [1, 0, 2]].copy()), persistent=False)
        self.register_buffer('right_faces', torch.tensor(faces.copy()), persistent=False)
        self.register_buffer('left_color', torch.tensor(HAND_COLOR_L).float() / 255.0, persistent=False)
        self.register_buffer('right_color', torch.tensor(HAND_COLOR_R).float() / 255.0, persistent=False)
        self.register_buffer('left_mask_color', torch.tensor([1, 0, 0]).float(), persistent=False)
        self.register_buffer('right_mask_color', torch.tensor([0, 0, 1]).float(), persistent=False)

        if right_mano_uv is not None and right_mano_uvfaces is not None:
            self.register_buffer('left_faceuv', torch.from_numpy(right_mano_uvfaces[..., [1, 0, 2]].copy()))
            self.register_buffer('right_faceuv', torch.from_numpy(right_mano_uvfaces.copy()))
            self.register_buffer('uv', torch.from_numpy(right_mano_uv).float())

        self.soft_rasterizer = self.build_rasterizer(self.img_size, soft=True)
        self.hard_rasterizer = self.build_rasterizer(self.img_size, soft=False)
        self.mask_shader = SoftSilhouetteShader()
        self.hard_shader = HardPhongShader()
        self.soft_shader = SoftPhongShader()

    def build_texture(self,
                      uv_verts: Optional[torch.Tensor] = None,
                      uv_faces: Optional[torch.Tensor] = None,
                      texture: Optional[torch.Tensor] = None,
                      v_color: Optional[torch.Tensor] = None):

        textures = super().build_texture(uv_verts, uv_faces, texture, v_color)
        if uv_verts is not None and uv_faces is not None and texture is not None:
            '''
            uv_verts: bs x N x 2
            uv_faces: bs x F x 2
            texture: bs x H x W x D
            '''
            bs = uv_verts.shape[0]
            f_num = uv_faces.shape[1]
            v_num = (f_num // self.left_faces.shape[0]) * 778
        elif v_color is not None:
            '''
            v_color: bs x N x D
            '''
            bs = v_color.shape[0]
            v_num = v_color.shape[1]
            f_num = (v_num // 778) * self.left_faces.shape[0]
        textures._num_faces_per_mesh = [f_num for _ in range(bs)]
        textures._num_verts_per_mesh = [v_num for _ in range(bs)]
        textures.valid = torch.ones((bs,), dtype=torch.bool, device=self.device())
        return textures

    def build_camera(self, cameras: Optional[torch.Tensor] = None,
                     scale: Optional[torch.Tensor] = None, trans2d: Optional[torch.Tensor] = None):
        cameras = super().build_camera(cameras, self.img_size, scale, trans2d)
        return cameras

    def visibility(self, fragments, v3d, K):
        zbuf = fragments.zbuf

        img_size = self.img_size
        v2d = torch.matmul(v3d, K.transpose(-1, -2))
        v2d = v2d[..., :2] / v2d[..., 2:]
        v2d = v2d * 2 / (img_size - 1) - 1
        inside = (v2d[..., 0] > -1) & (v2d[..., 0] < 1) & (v2d[..., 1] > -1) & (v2d[..., 1] < 1)
        depth = F.grid_sample(zbuf.permute(0, 3, 1, 2), v2d.unsqueeze(2),
                              align_corners=True,
                              padding_mode='zeros')[..., 0]
        depth = depth.permute(0, 2, 1).squeeze(-1)
        depth = depth * inside
        vis = torch.abs(depth - v3d[..., 2]) < 5e-3  # 5 mm
        return vis  # bs x N

    @torch.no_grad()
    def visibility2(self, fragments, v3d, faces, K):
        bs, N, _ = v3d.shape
        Fnum, _ = faces.shape
        bs, H, W, _ = fragments.pix_to_face.shape

        v2d = torch.matmul(v3d, K.transpose(-1, -2))
        v2d = v2d[..., :2] / v2d[..., 2:]
        v2d = v2d.long().clamp(min=0, max=self.img_size - 1)  # bs x N x 2
        v2d = v2d[..., 0] + v2d[..., 1] * W  # bs x N
        pix_to_face = fragments.pix_to_face[..., 0].reshape(bs, H * W)  # bs x H*W
        faceId = torch.gather(pix_to_face, dim=1, index=v2d)  # bs * N
        faceId = faceId - torch.arange(bs).to(faceId).unsqueeze(-1) * Fnum  # bs * N
        faceId[faceId < 0] = 0

        vertsId = faces[faceId.reshape(-1)].reshape(bs, N, 3)
        vid = torch.arange(N).to(vertsId).expand(bs, -1)
        vis = (vertsId[..., 0] == vid) | (vertsId[..., 1] == vid) | (vertsId[..., 2] == vid)
        return vis  # bs x N

    def proj(self, v3d, K):
        v2d = torch.matmul(v3d, K.transpose(-1, -2))
        v2d = v2d[..., :2] / v2d[..., 2:]
        return v2d

    def render_flow(self,
                    verts_left_0, verts_right_0, K0,
                    verts_left_1, verts_right_1, K1):
        cameras_0 = self.build_camera(cameras=K0)
        cameras_1 = self.build_camera(cameras=K1)
        rasterizer = self.hard_rasterizer

        verts_0 = torch.cat([verts_left_0, verts_right_0], dim=1)  # bs x N x 3
        verts_1 = torch.cat([verts_left_1, verts_right_1], dim=1)  # bs x N x 3
        faces = torch.cat([self.left_faces, self.right_faces + 778], dim=0)  # F x 3

        mesh_0 = Meshes(verts=verts_0, faces=faces.expand(verts_0.shape[0], -1, -1))
        mesh_1 = Meshes(verts=verts_1, faces=faces.expand(verts_1.shape[0], -1, -1))
        fragments_0 = rasterizer(mesh_0, cameras=cameras_0)
        fragments_1 = rasterizer(mesh_1, cameras=cameras_1)

        attributes_0 = verts_0[:, faces]  # bs x F x 3 x D
        attributes_1 = verts_1[:, faces]  # bs x F x 3 x D
        attributes = torch.cat([attributes_0, attributes_1], dim=-1)  # bs x F x 3 x 2*D

        valid_0 = fragments_0.pix_to_face[..., 0] >= 0
        valid_1 = fragments_1.pix_to_face[..., 0] >= 0
        pixel_vals_0 = self.render_face_attr(fragments_0, attributes)
        pixel_vals_1 = self.render_face_attr(fragments_1, attributes)
        bs, H, W = pixel_vals_0.shape[:-1]

        ori_0 = pixel_vals_0[..., :3]
        pre_0 = pixel_vals_0[..., 3:]
        pre_1 = pixel_vals_1[..., :3]
        ori_1 = pixel_vals_1[..., 3:]

        ori_0 = self.proj(ori_0.reshape(bs, H * W, 3), K0).reshape(bs, H, W, 2)
        pre_0 = self.proj(pre_0.reshape(bs, H * W, 3), K0).reshape(bs, H, W, 2)
        ori_1 = self.proj(ori_1.reshape(bs, H * W, 3), K1).reshape(bs, H, W, 2)
        pre_1 = self.proj(pre_1.reshape(bs, H * W, 3), K1).reshape(bs, H, W, 2)

        flow_0to1 = pre_0 - ori_0  # bs x H x W x 2
        flow_1to0 = pre_1 - ori_1  # bs x H x W x 2

        flow_0to1[~valid_0] = 0.0
        flow_1to0[~valid_1] = 0.0
        return flow_0to1, flow_1to0

    def render(self,
               verts_left: Optional[torch.Tensor],
               verts_right: Optional[torch.Tensor],
               cam_K: Optional[torch.Tensor] = None,
               scale: Optional[torch.Tensor] = None,
               trans2d: Optional[torch.Tensor] = None,
               output: List[str] = [],
               v_color: Optional[torch.Tensor] = None,  # bs x 2*N x D
               texture: Optional[torch.Tensor] = None,  # bs x H x W x 3
               soft: bool = False,
               ):
        result = {}
        for k in output:
            assert k in ['depth', 'rgb', 'mask', 'v_color', 'texture']

        if soft:
            rasterizer = self.soft_rasterizer
            shader = self.soft_shader
        else:
            rasterizer = self.hard_rasterizer
            shader = self.hard_shader

        if verts_left is None:
            verts_left = torch.zeros_like(verts_right)
        if verts_right is None:
            verts_right = torch.zeros_like(verts_left)

        bs = verts_left.shape[0]
        verts = torch.cat([verts_left, verts_right], dim=1)  # bs x N x 3
        faces = torch.cat([self.left_faces, self.right_faces + verts_left.shape[1]], dim=0)  # F x 3

        mesh = Meshes(verts=verts, faces=faces.expand(verts.shape[0], -1, -1))
        cameras = self.build_camera(cameras=cam_K, scale=scale, trans2d=trans2d)
        fragments = rasterizer(mesh, cameras=cameras)
        result['fragments'] = fragments
        result['alpha'] = (fragments.pix_to_face[..., 0] != -1).float()  # bs x H x W

        self.amblights.to(self.device())
        self.point_lights.to(self.device())
        shader.materials.to(self.device())

        # if 'vis' in output:
        #     assert not soft
        #     # vis = self.visibility(fragments, verts, cam_K)
        #     vis = self.visibility2(fragments, verts, faces, cam_K)
        #     result['vis_left'] = vis[:, :778]  # B x 778
        #     result['vis_right'] = vis[:, 778:]  # B x 778
        if 'depth' in output:
            result['depth'] = fragments.zbuf[..., 0]  # B x H x W
        if 'rgb' in output:
            color = torch.cat([self.left_color.expand_as(verts_left),
                               self.right_color.expand_as(verts_right)], dim=1)
            mesh.textures = self.build_texture(v_color=color)
            image = shader(fragments, mesh, lights=self.point_lights, cameras=cameras)
            result['rgb'] = (image[..., :3] * image[..., 3:4]).permute(0, 3, 1, 2)  # bs x 3 x H x W (0~1, bgr)
        if 'mask' in output:
            color = torch.zeros_like(verts[..., :2])  # bs x N x 2
            color[:, :verts.shape[1] // 2, 0] = 1.0
            color[:, verts.shape[1] // 2:, 1] = 1.0
            mask = self.render_face_attr(fragments, attributes=color[:, faces])  # bs x H x W x 2
            result['mask'] = torch.cat([1 - result['alpha'].unsqueeze(-1), mask], dim=-1).permute(0, 3, 1,
                                                                                                  2)  # bs x 3 x H x W (bg, left, right)
        if 'v_color' in output and v_color is not None:
            result['v_color'] = self.render_face_attr(fragments, attributes=v_color[:, faces]).permute(0, 3, 1,
                                                                                                       2)  # bs x D x H x W
        if 'texture' in output and texture is not None and hasattr(self, 'uv'):
            uv = self.uv.expand(bs, -1, -1)  # bs x N2 x 2
            faceuv = torch.cat([self.left_faceuv, self.right_faceuv], dim=0).expand(bs, -1, -1)  # bs x F x 3
            mesh.textures = self.build_texture(uv_verts=uv, uv_faces=faceuv, texture=texture)
            image = shader(fragments, mesh, lights=self.point_lights, cameras=cameras)
            result['texture'] = (image[..., :3] * image[..., 3:4]).permute(0, 3, 1, 2)  # bs x 3 x H x W
        return result


def vis_depth(depth):
    '''
    depth: bs x H x W
    '''
    bs, H, W = depth.shape
    d_flat = depth.reshape(bs, -1).detach().clone()
    d_max = torch.max(d_flat, dim=-1)[0]  # bs
    d_flat[d_flat < 0] = d_max.max()
    d_min = torch.min(d_flat, dim=-1)[0]  # bs
    d_max = d_max.unsqueeze(-1).unsqueeze(-1) + 0.01
    d_min = d_min.unsqueeze(-1).unsqueeze(-1) - 0.02
    depth = torch.clamp(depth, d_min, d_max)
    depth = (depth - d_min) / (d_max - d_min)
    return depth


class ManoRenderer(ManoRenderer_twohand):
    def __init__(self, img_size: int, right_mano_faces: np.ndarray,
                 right_mano_uv: Optional[np.ndarray] = None, right_mano_uvfaces: Optional[np.ndarray] = None):
        super().__init__(img_size, right_mano_faces, right_mano_uv, right_mano_uvfaces)

        cam_R = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
        cam_T = torch.tensor([0, 0.08, 1.2]).float()
        f = 0.8 * img_size / 2 / 0.085
        cam_K = torch.tensor([[f, 0, img_size / 2], [0, f, img_size / 2], [0, 0, 1]]).float()

        self.register_buffer('cam_R', cam_R)
        self.register_buffer('cam_T', cam_T)
        self.register_buffer('cam_K', cam_K)

    def render(self,
               verts_right: torch.Tensor,
               verts_left: Optional[torch.Tensor] = None,
               cam_K: Optional[torch.Tensor] = None,
               cam_R: Optional[torch.Tensor] = None,
               cam_T: Optional[torch.Tensor] = None,
               scale: Optional[torch.Tensor] = None,
               trans2d: Optional[torch.Tensor] = None,
               zero_RT: bool = False) -> (torch.Tensor, torch.Tensor):
        if zero_RT:
            cam_K = self.cam_K.expand(verts_right.shape[0], -1, -1)  # bs x 3 x 3
            cam_R = self.cam_R  # 3 x 3
            cam_T = self.cam_T  # 3
        # else:
        #    assert cam_K is not None

        if cam_R is not None:
            verts_right = torch.matmul(verts_right, cam_R.transpose(-1, -2))
        if cam_T is not None:
            verts_right = verts_right + cam_T.unsqueeze(1)

        out = super().render(verts_left=verts_left,
                             verts_right=verts_right,
                             cam_K=cam_K,
                             scale=scale, trans2d=trans2d, output=['rgb'])  # bs x 3 x H x W
        img = out['rgb']  # bs x 3 x H x W
        mask = out['alpha'].unsqueeze(1)  # bs x 1 x H x W

        return img, mask

    def render_twohand_orth(self, scale_left=None, trans2d_left=None,
                            scale_right=None, trans2d_right=None,
                            v3d_left=None, v3d_right=None):

        scale = scale_left
        trans2d = trans2d_left

        s = scale_right / scale_left
        d = -(trans2d_left - trans2d_right) / 2 / scale_left.unsqueeze(-1)

        s = s.unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(1)
        v3d_right = s * v3d_right
        v3d_right[..., :2] = v3d_right[..., :2] + d

        # scale = (scale_left + scale_right) / 2
        # trans2d = (trans2d_left + trans2d_right) / 2

        out = super().render(scale=scale, trans2d=trans2d,
                             verts_left=v3d_left, verts_right=v3d_right, output=['rgb'])
        img = out['rgb']  # bs x 3 x H x W
        mask = out['alpha'].unsqueeze(1)  # bs x 1 x H x W

        return img, mask