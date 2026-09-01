import os
import json
from operator import itemgetter
import bisect
import csv
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from util import draw_umich_gaussian

import lightning as pl


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32) # [160, 250]
    reg_mask = np.zeros([top_k], dtype=np.int64)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


def get_headfoot_gt(Rshape, x_s, y_head, y_foot, v_s=None, reduce=4, kernel_size=4):
    H, W = Rshape
    head_map = np.zeros([1, H, W], dtype=np.float32)
    foot_map = np.zeros([1, H, W], dtype=np.float32)

    for k in range(len(v_s)):
        ct_head = np.array([x_s[k] / reduce, y_head[k] / reduce], dtype=np.float32)
        ct_foot = np.array([x_s[k] / reduce, y_foot[k] / reduce], dtype=np.float32)
        if 0 <= ct_head[0] < W and 0 <= ct_head[1] < H:
            ct_int_head = ct_head.astype(np.int32)
            draw_umich_gaussian(head_map[0], ct_int_head, kernel_size / reduce)
        if 0 <= ct_foot[0] < W and 0 <= ct_foot[1] < H:
            ct_int_foot = ct_foot.astype(np.int32)
            draw_umich_gaussian(foot_map[0], ct_int_foot, kernel_size / reduce)
    
    ret = {'head_map': torch.from_numpy(head_map), 'foot_map': torch.from_numpy(foot_map)}
    return ret


class GMVDSequenceDataset(VisionDataset):
    def __init__(
            self, 
            root,
            da3_root,
            train, 
            input_reduce=1.5,
            world_reduce=4,
            img_reduce=4,
            world_kernel_size=10, 
            img_kernel_size=10,
            top_k=100,
            train_ratio=1,
            sample_require=0,
        ):
        super().__init__(root)

        with open(os.path.join(self.root, 'config.json'), 'r') as f:
            config = json.load(f)

        if 'seq' in self.root.split('/')[-1]:
            self.__name__ = f'{self.root.split("/")[-3]}_{self.root.split("/")[-2]}_{self.root.split("/")[-1]}'
        else:
            self.__name__ = f'{self.root.split("/")[-1]}'
        
        self.gt_fname = os.path.join(self.root, 'gt.txt')
        self.da3_root = da3_root

        self.img_shape = config['img_shape']  # [1080, 1920]
        self.worldgrid_shape = [int(config['grid_shape'][0]), int(config['grid_shape'][1])] # [640, 1000]
        self.num_cam, self.num_frame = config['num_cam'], config['num_frames'] # 6, 400
        if sample_require:
            self.frame_step = int(self.num_frame // sample_require)
        else:
            self.frame_step = 1

        self.grid_cell, self.origin = config['grid_cell'], config['origin'] # 0.025, [0, 0]
        self.region_size = config['region_size'] # [16, 25]
        self.top_k = top_k

        self.input_reduce = input_reduce
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size

        self.transform = T.Compose([
            T.ToTensor(), 
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Resize((np.array(self.img_shape) // self.input_reduce).astype(int).tolist(), antialias=True)
        ])

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / (self.input_reduce * self.img_reduce)).astype(int).tolist()

        if train:
            frame_range = range(0, int(train_ratio*self.num_frame), self.frame_step)
        else:
            frame_range = range(int(train_ratio*self.num_frame), self.num_frame, self.frame_step)

        self.img_fpaths = self.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)
    
    def download(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))
    
    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            if camera_folder == '.DS_Store':
                continue
            if camera_folder.split('.')[-1] == 'mp4':
                continue
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths
    
    def get_worldgrid_from_pos(self, pos):
        R, C = self.worldgrid_shape
        grid_x = pos % C
        grid_y = pos // C
        return np.array([grid_x, grid_y], dtype=int)
    
    def __len__(self):
        return len(self.world_gt.keys())
    
    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]

        imgs, imgs_gt = [], []
        for cam in range(self.num_cam):
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            imgs.append(self.transform(img))
            
            img_x_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2
            img_y_head = img_bboxs[:, 1]
            img_y_foot = img_bboxs[:, 3]

            img_gt = get_headfoot_gt(
                self.Rimg_shape, img_x_s, img_y_head, img_y_foot, v_s=img_pids,
                reduce=int(self.input_reduce * self.img_reduce), kernel_size=self.img_kernel_size
            )
            imgs_gt.append(img_gt)
        
        imgs = torch.stack(imgs)
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}

        world_pt_s, world_pid_s = self.world_gt[frame]
        world_gt = get_gt(self.Rworld_shape, world_pt_s[:, 0], world_pt_s[:, 1], v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        
        da3_output = np.load(os.path.join(self.da3_root, f'{frame:04}.npz'))
        da3_point = torch.from_numpy(da3_output['point']).to(torch.float32)
        da3_feat = np.stack(
            (
                da3_output['feat_layer_19'],
                da3_output['feat_layer_27'],
                da3_output['feat_layer_33'],
                da3_output['feat_layer_39']
            ), axis=0
        )
        da3_feat = torch.from_numpy(da3_feat)

        return imgs, da3_feat, da3_point, world_gt, imgs_gt, frame, \
            torch.tensor(self.region_size), torch.tensor(self.Rworld_shape)


class GMVDDataModule(pl.LightningDataModule):
    def __init__(
            self,
            csv_path: str,
            data_dir: str,
            da3_dir: str,
            batch_size: int = 1,
            num_workers: int = 4,
            input_reduce: float = 1.5,
            world_reduce: int = 4,
            img_reduce: int = 4,
            world_kernel_size: int = 10,
            img_kernel_size: int = 10, 
            accumulate_grad_batches=16,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.da3_dir = da3_dir
        self.batch_size = batch_size
        assert self.batch_size == 1
        self.num_workers = num_workers
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.world_reduce = world_reduce
        self.input_reduce = input_reduce
        self.img_reduce = img_reduce
        self.world_kernel_size = world_kernel_size
        self.img_kernel_size = img_kernel_size

        self.data_train = None
        self.data_val = None
        self.data_test = None
    
    def setup(self, stage: Optional[str] = None):
        dataset_list = []

        if stage == 'fit':
            f = open(self.csv_path)
            data_path = csv.reader(f)
            for i, data_row in enumerate(data_path):
                train_ratio = float(data_row[2])
                sample_require = int(data_row[3])
                path = os.path.join(self.data_dir, str(data_row[1])[12:])

                if str(data_row[1]).split('/')[2] == 'DATASETS':
                    da3_path = os.path.join(self.da3_dir, str(data_row[1])[21:])
                elif str(data_row[1]).split('/')[2] == 'detect':
                    da3_path = os.path.join(self.da3_dir, str(data_row[1])[19:])
                else:
                    NotImplementedError

                if data_row[0] == 'train':
                    dataset_obj = GMVDSequenceDataset(
                        root=path,
                        da3_root=da3_path,
                        train=True,
                        input_reduce=self.input_reduce,
                        world_reduce=self.world_reduce,
                        img_reduce=self.img_reduce,
                        world_kernel_size=self.world_kernel_size,
                        img_kernel_size=self.img_kernel_size,
                        train_ratio=train_ratio,
                        sample_require=sample_require,
                    )
                    break
        
        elif stage == 'test':
            f = open(self.csv_path)
            data_path = csv.reader(f)
            for i, data_row in enumerate(data_path):
                train_ratio = float(data_row[2])
                sample_require = int(data_row[3])
                path = os.path.join(self.data_dir, str(data_row[1])[12:])

                if str(data_row[1]).split('/')[2] == 'DATASETS':
                    da3_path = os.path.join(self.da3_dir, str(data_row[1])[21:])
                elif str(data_row[1]).split('/')[2] == 'detect':
                    da3_path = os.path.join(self.da3_dir, str(data_row[1])[19:])
                else:
                    NotImplementedError

                dataset_obj = GMVDSequenceDataset(
                        root=path,
                        da3_root=da3_path,
                        train=False,
                        input_reduce=self.input_reduce,
                        world_reduce=self.world_reduce,
                        img_reduce=self.img_reduce,
                        world_kernel_size=self.world_kernel_size,
                        img_kernel_size=self.img_kernel_size, 
                        train_ratio=train_ratio,
                        sample_require=sample_require,
                    )
                break
        
        if stage == 'fit':
            print("Training datasets")
            self.data_train = dataset_obj
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )