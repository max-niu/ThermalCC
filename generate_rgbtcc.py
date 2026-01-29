import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage import gaussian_filter

import xml.etree.ElementTree as ET
import json

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def get_points(label):
    tree = ET.parse(label)
    root = tree.getroot()
    points = [[int(obj.find('point').find('x').text), int(obj.find('point').find('y').text)]for obj in root.findall('object')]
    return points
    

'''change your path'''
root = './rgbtcc'

part_A_train = os.path.join(root, 'train')
part_A_test = os.path.join(root, 'test')



path_sets = [part_A_train, part_A_test]

if not os.path.exists(part_A_train.replace('train', 'train_gt_fidt_map')):
    os.makedirs(part_A_train.replace('train', 'train_gt_fidt_map'))

if not os.path.exists(part_A_test.replace('test', 'test_gt_fidt_map')):
    os.makedirs(part_A_test.replace('test', 'test_gt_fidt_map'))

if not os.path.exists(part_A_train.replace('train', 'train_gt_show')):
    os.makedirs(part_A_train.replace('train', 'train_gt_show'))

if not os.path.exists(part_A_test.replace('test', 'test_gt_show')):
    os.makedirs(part_A_test.replace('test', 'test_gt_show'))


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*_T.jpg')):
        img_paths.append(img_path)

img_paths.sort()


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map


for img_path in img_paths:
    print(img_path)
    Img_data = cv2.imread(img_path)


    gt_file = img_path.replace('_T.jpg', '_GT.json')
    points = read_json_file(gt_file)['points']
    Gt_data = np.asarray(points)

    fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    
    h5py_file = img_path.replace('.jpg', '.h5').replace('train', 'train_gt_fidt_map') if '/train/' in img_path else img_path.replace('.jpg', '.h5').replace('test', 'test_gt_fidt_map')
    with h5py.File(h5py_file, 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint

    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

    '''for visualization'''
    write_file = img_path.replace('train', 'train_gt_show').replace('jpg', 'jpg') if '/train/' in img_path else img_path.replace('test', 'test_gt_show').replace('jpg', 'jpg')
    cv2.imwrite(write_file, fidt_map1)
