import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# from lib.utils.coarsening import build_graph

class meshLinearUpsampler(nn.Module):
    def __init__(self, weight_list):
        super().__init__()
        self.layer_num = len(weight_list)
        self.vertices_num = []
        for i in range(self.layer_num):
            self.vertices_num.append(weight_list[i].shape[1])
        self.vertices_num.append(weight_list[-1].shape[0])

        to_mesh_weight = [weight_list[i].copy() for i in range(self.layer_num)]
        for i in range(self.layer_num):
            for j in range(i + 1, self.layer_num):
                to_mesh_weight[i] = weight_list[j] @ to_mesh_weight[i]

        for i in range(self.layer_num):
            self.register_buffer('up_{}'.format(i), torch.from_numpy(weight_list[i]).float(), persistent=False)
            self.register_buffer('to_mesh_{}'.format(i), torch.from_numpy(to_mesh_weight[i]).float(), persistent=False)

    def upsample(self, x):
        # x: bs x N x 3
        for i in range(self.layer_num):
            if x.shape[1] == self.vertices_num[i]:
                weight = getattr(self, 'up_{}'.format(i))
                return torch.matmul(weight, x)
        raise 'wrong vertices number!'

    def to_mesh(self, x):
        # x: bs x N x 3
        for i in range(self.layer_num):
            if x.shape[1] == self.vertices_num[i]:
                weight = getattr(self, 'to_mesh_{}'.format(i))
                return torch.matmul(weight, x)
        raise 'wrong vertices number!'

    def forward(self, x):
        return self.to_mesh(x)


class meshCoarseSampler(nn.Module):
    def __init__(self, vertex_num=1, graph_perm_reverse=[0], graph_perm=[0]):
        super().__init__()
        self.vNum = vertex_num
        self.gcnNum = len(graph_perm)

        self.graph_perm_reverse = torch.tensor(graph_perm_reverse[:vertex_num], dtype=torch.long)
        self.graph_perm = torch.tensor(graph_perm, dtype=torch.long)

        self.register_buffer('gcn2v_idx', self.graph_perm_reverse, persistent=False)
        self.register_buffer('v2gcn_idx', self.graph_perm, persistent=False)

    def upsample(self, x, p=2):
        return graph_upsample(x, p)

    def downsample(self, x, p=2, method='avg'):
        if method == 'avg':
            return graph_avg_pool(x, p)
        elif method == 'max':
            return graph_max_pool(x, p)
        else:
            raise "wrong downsample type"

    def GCN2MESH(self, x):
        p = self.gcnNum // x.shape[1]
        if p > 1:
            x = self.upsample(x, p)
        return torch.index_select(x, 1, self.gcn2v_idx)

    def MESH2GCN(self, x):
        return torch.index_select(x, 1, self.v2gcn_idx)

    def convert(self, x, out_vNum, method='avg'):
        if x.shape[1] == self.vNum:
            x = self.MESH2GCN(x)
        in_vNum = x.shape[1]

        if out_vNum == self.vNum:
            out_vNum = self.gcnNum
            out_mesh = True
        else:
            out_mesh = False

        if in_vNum > out_vNum:
            x = self.downsample(x, in_vNum // out_vNum, method)
        elif in_vNum < out_mesh:
            x = self.upsample(x, out_vNum // in_vNum)
        else:
            x = x

        if out_mesh:
            x = self.GCN2MESH(x)
        return x


class meshPartSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.part_num = len(config)
        weight = np.zeros((self.part_num, 778))
        for i in range(self.part_num):
            weight[i, config[i]['verts_id']] = 1
        part2mesh = weight.T
        mesh2part = weight
        part2mesh = part2mesh / np.sum(part2mesh, axis=1, keepdims=True)  # 778 x 18
        mesh2part = mesh2part / np.sum(mesh2part, axis=1, keepdims=True)  # 18 x 778

        self.register_buffer('part2mesh', torch.from_numpy(part2mesh).float(), persistent=False)
        self.register_buffer('mesh2part', torch.from_numpy(mesh2part).float(), persistent=False)

    def MESH2PART(self, x):
        x = torch.matmul(self.mesh2part, x)  # bs x 18 x f
        return x

    def PART2MESH(self, x):
        x = torch.matmul(self.part2mesh, x)  # bs x 778 x f
        return x


class meshSampler(nn.Module):
    def __init__(self, upsample_weight_list,
                 vertex_num, graph_perm_reverse, graph_perm,
                 part_config):
        super().__init__()
        self.linear_layer = meshLinearUpsampler(upsample_weight_list)
        self.coarse_layer = meshCoarseSampler(vertex_num, graph_perm_reverse, graph_perm)
        self.part_layer = meshPartSampler(part_config)

    def linear_upsample(self, x):
        return self.linear_layer.upsample(x)

    def linear_tomesh(self, x):
        return self.linear_layer.to_mesh(x)

    def coarsen_gcn2mesh(self, x):
        return self.coarse_layer.GCN2MESH(x)

    def coarsen_mesh2gcn(self, x):
        return self.coarse_layer.MESH2GCN(x)

    def coarsen_upsample(self, x, p=2):
        return self.coarse_layer.upsample(x, p)

    def coarsen_downsample(self, x, p=2, method='avg'):
        return self.coarse_layer.downsample(x, p, method)

    def coarsen_convert(self, x, out_vNum, method='avg'):
        return self.coarse_layer.convert(x, out_vNum, method)

    def part_mesh2part(self, x):
        return self.part_layer.MESH2PART(x)

    def part_part2mesh(self, x):
        return self.part_layer.PART2MESH(x)

    def part_gcn2part(self, x):
        x = self.linear_tomesh(x)
        return self.part_mesh2part(x)

    def part_part2gcn(self, x, out_vNum, method='avg'):
        x = self.part_part2mesh(x)
        return self.coarsen_convert(x, out_vNum, method)


def get_meshsample_layer(graph_path='/workspace/hamer_twohand/hamer_finetune/misc/mano/mano_graph_dict.pkl',
                         up_sample_path='/workspace/hamer_twohand/hamer_finetune/misc/mano/mano_upsample.pkl',
                         seg_path='/workspace/hamer_twohand/hamer_finetune/misc/mano/manoPart.pkl') -> meshSampler:
    with open(graph_path, 'rb') as file:
        graph_dict = pickle.load(file)
    with open(up_sample_path, 'rb') as file:
        upsample_weight = pickle.load(file)
    with open(seg_path, 'rb') as file:
        part_config = pickle.load(file)
    return meshSampler(upsample_weight,
                       778,
                       graph_dict['graph_perm_reverse'],
                       graph_dict['graph_perm'],
                       part_config['config18'])


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
    return L


def graph_max_pool(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.MaxPool1d(p)(x)  # B x F x V/p
        x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
        return x
    else:
        return x


def graph_avg_pool(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.AvgPool1d(p)(x)  # B x F x V/p
        x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
        return x
    else:
        return x


def graph_upsample(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x


# def get_next_ring(neigh_list, last_ring, last_disk):
#     ring = []
#     for i in last_ring:
#         for j in neigh_list[i]:
#             if j not in last_disk and j not in ring:
#                 ring.append(j)
#     return ring


# def get_one_spiral(idx, neigh_list, seq_length, dilation=1, coord=None):
#     if len(neigh_list[idx]) == 0:
#         return [idx for i in range(seq_length)]

#     spiral = [idx]
#     last_ring = [idx]
#     while(len(spiral) < seq_length * dilation):
#         next_ring = get_next_ring(neigh_list, last_ring, spiral)
#         spiral += next_ring
#         last_ring = next_ring

#     if coord is not None:
#         x = coord[idx]
#         y = coord[spiral]
#         dist = np.linalg.norm(x - y, axis=-1)
#         idx = np.argsort(dist, axis=0).tolist()
#         spiral = [spiral[idx[i]] for i in range(len(idx))]

#     return spiral[:seq_length * dilation][::dilation]


# def extract_spirals(neigh_list, seq_length, dilation=1, coord=None):
#     spirals = []
#     for v in range(len(neigh_list)):
#         spirals.append(get_one_spiral(v, neigh_list, seq_length, dilation, coord))
#     return spirals


# def _build_graph(path, level):
#     data = pickle.load(open(path, 'rb'), encoding='latin1')
#     faces = data['f']
#     verts = data['v_template']
#     graph_dict = build_graph(faces, coarsening_levels=level)

#     sampler = meshCoarseSampler(vertex_num=verts.shape[0],
#                                 graph_perm_reverse=graph_dict['graph_perm_reverse'],
#                                 graph_perm=graph_dict['graph_perm'])
#     verts = torch.from_numpy(verts).unsqueeze(0)
#     verts = sampler.MESH2GCN(verts)

#     graph_dict['spirals'] = []
#     for i in range(len(graph_dict['coarsen_graphs_adj'])):
#         adj = graph_dict['coarsen_graphs_adj'][i].tocoo()
#         x = adj.row
#         y = adj.col
#         neigh = [[] for i in range(adj.shape[0])]
#         for i in range(x.shape[0]):
#             neigh[x[i]].append(y[i])
#         spirals = extract_spirals(neigh, seq_length=32, dilation=1, coord=sampler.downsample(verts, p=verts.shape[1] // adj.shape[0])[0])
#         graph_dict['spirals'].append(spirals)

#     return graph_dict


# def build_mano_graph(mano_path, save_path):
#     if not os.path.exists(save_path):
#         graph_dict = _build_graph(mano_path, level=4)

#         with open(save_path, 'wb') as file:
#             pickle.dump(graph_dict, file)
