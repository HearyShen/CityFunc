import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import json
import os
from PIL import Image
import pandas as pd
import numpy as np
import datetime

class CityFuncDataset(Dataset):
    """ City Func Datasets
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_root = os.path.join(root, "image")
        self.visit_root = os.path.join(root, "visit")
        self.idx2visit = os.listdir(self.visit_root)

        # 用字典查询代替类型转换，可以减少一部分计算时间
        self.date2position = {}
        self.datestr2dateint = {}
        self.str2int = {}
        for i in range(24):
            self.str2int[str(i).zfill(2)] = i

        # 访问记录内的时间从2018年10月1日起，共182天
        # 将日期按日历排列
        for i in range(182):
            date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
            date_int = int(date.__str__().replace("-", ""))
            self.date2position[date_int] = [i%7, i//7]
            self.datestr2dateint[str(date_int)] = date_int

    def __len__(self):
        '''
        Return the count of the samples in this dataset
        '''
        return len(os.listdir(self.visit_root))

    def __getitem__(self, idx):
        '''
        Return the idx th image, visit record and label
        '''
        visit_filename = self.idx2visit[idx]
        imgname = os.path.splitext(visit_filename)[0] + ".jpg"
        class_label = visit_filename[7:10]

        # image
        image = Image.open(os.path.join(self.img_root, class_label, imgname)).convert('RGB')
        image = self.transforms(image)

        # visit
        visit_path = os.path.join(self.visit_root, visit_filename)
        visit_numpy_cache_path = os.path.join(self.visit_root, os.path.splitext(visit_filename)[0]+".npy")
        if os.path.exists(visit_numpy_cache_path):
            visit = np.load(visit_numpy_cache_path)
        else:
            visit = self.visit2array(visit_path)    # a 7×26×24 numpy matrix
            np.save(visit_numpy_cache_path, visit)
        visit = transforms.ToTensor()(visit).float()      # a 24×7×26 tensor
        # visit = torch.from_numpy(visit.transpose(2,1,0)).float()
        
        # label
        label = int(class_label) - 1    # label in PyTorch starts from 0 (rather than 1)

        return (image, visit), label

    def visit2array(self, visit_path):
        visit_table = pd.read_csv(visit_path, header=None, sep='\t')
        visit_recordings = visit_table[1]        # without visitors USERID column
        init = np.zeros((7, 26, 24))
        for recording in visit_recordings:
            temp = []
            for item in recording.split(','):
                temp.append([item[0:8], item[9:].split("|")])
            for date, visit_lst in temp:
                # x - 星期几（0~6）
                # y - 第几周（0~25）
                # z - 几点钟（0~23）
                # value - 到访的总人数
                x, y = self.date2position[self.datestr2dateint[date]]
                for visit in visit_lst: # 统计到访的总人数
                    init[x][y][self.str2int[visit]] += 1
        return init


