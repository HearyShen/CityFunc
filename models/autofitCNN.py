import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AutoFitNet(nn.Module):
    def __init__(self, arch, pretrained=False, num_classes=1000):
        super(AutoFitNet, self).__init__()
        # create model
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            self.model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            self.model = models.__dict__[arch]()

        

    def forward(self, x):
        x = self.model(x)

        return x
    
