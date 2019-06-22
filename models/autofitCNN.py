import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ImageFitNet(nn.Module):
    def __init__(self):
        super(ImageFitNet, self).__init__()
        self.model = models.resnet34(num_classes=256)

    def forward(self, x):
        out = self.model(x)
        return out

class VisitFitNet(nn.Module):
    def __init__(self):
        super(VisitFitNet, self).__init__()
        self.conv0 = nn.Conv2d(24, 64, 3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.downsample128 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.downsample256 = nn.Conv2d(128, 256, 1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        out = self.conv0(x) # x: batch_size×24×7×26
        identity = out      # out: batch_size×64×7×26

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out) # out: batch_size×64×4×13

        out = self.downsample128(out)

        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out) # out: batch_size×128×3×8

        out = self.downsample256(out)

        identity = out
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out) # out: batch_size×256×2×5

        out = self.avgpool(out) # out: batch_size×256×1×1

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)      # out: batch_size×256

        return out


class AutoFitNet(nn.Module):
    def __init__(self, arch, pretrained=False, num_classes=1000):
        super(AutoFitNet, self).__init__()
        # init models
        self.image_model = ImageFitNet()
        self.visit_model = VisitFitNet()
        self.fc = nn.Linear(512, 9) 

    def forward(self, x):
        # image: batch_size×3×224×224
        # visit: batch_size×24×7×26
        (image, visit) = x
        image_out = self.image_model(image)
        visit_out = self.visit_model(visit)
        out = torch.cat((image_out, visit_out), 1)
        out = self.fc(out)

        return F.softmax(out, dim=1)
    
