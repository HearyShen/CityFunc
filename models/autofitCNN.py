import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageFitNet(nn.Module):
    def __init__(self):
        super(ImageFitNet, self).__init__()
        self.model = models.resnet34(num_classes=9)

    def forward(self, x):
        out = self.model(x)
        return out

class VisitFitNet(nn.Module):
    def __init__(self):
        super(VisitFitNet, self).__init__()
        pass

    def forward(self, x):
        return None


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
    
