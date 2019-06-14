from torch.utils.data import Dataset
import torchvision
# import torchvision.transforms as transforms
import json
import os
from PIL import Image
import os

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
        visit = None   # @todo

        label = int(class_label) - 1    # label in PyTorch starts from 0 (rather than 1)
        
        return image, visit, label
