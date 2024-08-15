import torch
import os
import sys
import pickle
import json
import numpy as np
import cv2 as cv
import random
import tqdm
import pickle

class Twohand_Test(torch.utils.data.Dataset):
    def __init__(self,
                 data_path = '/workspace/twohand_crop/hard_cases',
                 test_path = '/workspace/twohand_crop/test_cases'):
        self.data_path = data_path
        self.test_path = test_path
        with open(os.path.join(data_path,'bbox.json'),'r') as f:
            self.bbox_data = json.load(f)

        with open(os.path.join(test_path,'bbox.json'),'r') as f:
            self.bbox_test = json.load(f)

        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        self.MEAN = 255. * np.array(IMAGE_MEAN)
        self.STD = 255. * np.array(IMAGE_STD)
        self.img_size = 256

    def __len__(self):
        return len(self.bbox_data) * 100

    def __getitem__(self, item):
        if random.random() > 0.75:
            bbox_data = self.bbox_test
            data_path = self.test_path
        else:
            bbox_data = self.bbox_data
            data_path = self.data_path
        index = random.randint(0,len(bbox_data)-1)
        bboxes = bbox_data[f'{index}']
        output = {}
        output['bbox_left'] = torch.FloatTensor(bboxes['bbox_left'])
        output['bbox_right'] = torch.FloatTensor(bboxes['bbox_right'])
        output['bbox_merge'] = torch.FloatTensor(bboxes['bbox_merge'])

        left_path = os.path.join(data_path,'left',f'left_{index}.jpg')
        right_path = os.path.join(data_path, 'right', f'right_{index}.jpg')
        inter_path = os.path.join(data_path, 'inter', f'inter_{index}.jpg')

        img_left = cv.imread(left_path)

        img_left = cv.resize(img_left, (self.img_size, self.img_size)).astype(np.float32)
        for n_c in range(3):
            img_left[:, :, n_c] = (img_left[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
        img_left = torch.from_numpy(img_left).float()  # H x W x 3
        output['img_left'] = img_left

        img_right = cv.imread(right_path)
        img_right = cv.resize(img_right, (self.img_size, self.img_size)).astype(np.float32)
        for n_c in range(3):
            img_right[:, :, n_c] = (img_right[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
        img_right = torch.from_numpy(img_right).float()  # H x W x 3
        output['img_right'] = img_right

        img = cv.imread(inter_path)
        img = cv.resize(img, (self.img_size, self.img_size)).astype(np.float32)
        #for n_c in range(3):
        #    img[:, :, n_c] = (img[:, :, n_c] - self.MEAN[n_c]) / self.STD[n_c]
        img = torch.from_numpy(img).float()  # H x W x 3
        output['img_inter'] = img
        output['index'] = index

        return output


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