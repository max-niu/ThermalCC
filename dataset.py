import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        if self.args['preload_data'] == True:
            fname = self.lines[index]['fname']
            img = self.lines[index]['img']
            kpoint = self.lines[index]['kpoint']
            fidt_map = self.lines[index]['fidt_map']
            
            gd_root = "./datasets/rgbtcc"
            gd_path = f"{gd_root}/train/{fname}".replace('_T.jpg', 'R.npy') if self.train else f"{gd_root}/test/{fname}".replace('_T.jpg', 'R.npy')
            
            gd_root = "./datasets/DroneRGBT"
            gd_path = f"{gd_root}/train/{fname}".replace('.jpg', '.npy') if self.train else f"{gd_root}/test/{fname}".replace('.jpg', '.npy')
            
            keypoints = np.load(gd_path)
            assert len(keypoints) > 0


        else:
            img_path = self.lines[index]
            fname = os.path.basename(img_path)
            img, fidt_map, kpoint = load_data_fidt(img_path, self.args, self.train)

        '''data augmention'''
        aug = False
        if self.train == True:
            if random.random() > 0.5:
                fidt_map = np.ascontiguousarray(np.fliplr(fidt_map))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.ascontiguousarray(np.fliplr(kpoint))
                aug = True


        fidt_map = fidt_map.copy()
        kpoint = kpoint.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        '''crop size'''
        if self.train == True:
            fidt_map = torch.from_numpy(fidt_map).cuda()

            width = self.args['crop_size']
            height = self.args['crop_size']
            
            pad_y = max(0, width - img.shape[1])
            pad_x = max(0, height - img.shape[2])
            if pad_y + pad_x > 0:
                img = F.pad(img, [0, pad_x, 0, pad_y], value=0)
                fidt_map = F.pad(fidt_map, [0, pad_x, 0, pad_y], value=0)
                kpoint = np.pad(kpoint, [(0, pad_y), (0, pad_x)], mode='constant', constant_values=0)
            # print(img.shape)
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            fidt_map = fidt_map[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            
            # ===
            i, j, h, w = (crop_size_x, crop_size_y, height, width)
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  # change coodinate
            
            if aug and len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]

            return fname, img, fidt_map, torch.from_numpy(kpoint.copy()), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), self.args['crop_size']

        return fname, img, fidt_map, torch.from_numpy(kpoint.copy())


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

