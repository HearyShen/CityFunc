from torch.utils.data import Dataset
import torchvision
# import torchvision.transforms as transforms
import json
import os
from PIL import Image
import pandas as pd
import numpy as np

class CityFuncDataset(Dataset):
    """ City Func Datasets
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_root = os.path.join(root, "image")
        self.visit_root = os.path.join(root, "visit")
        self.idx2visit = os.listdir(self.visit_root)

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

        image = Image.open(os.path.join(self.img_root, class_label, imgname)).convert('RGB')
        image = self.transforms(image)

        visit_path = os.path.join(self.visit_root, visit_filename)
        visit = self.visit2array(visit_path)

        visit = None   # @TODO

        label = int(class_label) - 1    # label in PyTorch starts from 0 (rather than 1)
        
        return image, visit, label

    def visit2array(self, visit_path):
        visit_table = pd.read_table(visit_path, header=None)
        visit_recordings = visit_table[1]        # without visitors USERID column
        init = np.zeros((7, 26, 24))
        for recording in visit_recordings:
            temp = []
            for item in recording.split(','):
                temp.append([item[0:8], item[9:].split("|")])
            for date, visit_lst in temp:
                # x - 第几周
                # y - 第几天
                # z - 几点钟
                # value - 到访的总人数
                x, y = date2position[datestr2dateint[date]]
                for visit in visit_lst: # 统计到访的总人数
                    init[x][y][str2int[visit]] += 1
        return init


