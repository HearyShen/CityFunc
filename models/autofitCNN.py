import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .visitFitNet import VisitFitNet


class ImageFitNet(nn.Module):
    def __init__(self):
        super(ImageFitNet, self).__init__()
        self.model = models.resnet34(num_classes=256)

    def forward(self, x):
        out = self.model(x)
        return out

class AutoFitNet(nn.Module):
    def __init__(self, arch, pretrained=False, num_classes=1000):
        super(AutoFitNet, self).__init__()
        # init models
        self.image_model = ImageFitNet()
        self.visit_model_24x7x26 = VisitFitNet(24, 128)
        self.visit_model_26x7x24 = VisitFitNet(26, 128)
        self.fc = nn.Linear(512, 9) 

    def forward(self, x):
        # image: batch_size×3×224×224
        # visit: batch_size×24×7×26
        (image, visit) = x
        visit_24x7x26 = visit
        visit_26x7x24 = visit.permute(0, 3, 2, 1)
        image_out = self.image_model(image)             # image_out: batch_size×256
        visit_out_weekdayAndWeek = self.visit_model_24x7x26(visit_24x7x26)     # visit_out_weekdayAndWeek: batch_size×128
        visit_out_weekdayAndHour = self.visit_model_26x7x24(visit_26x7x24)     # visit_out_weekdayAndHour: batch_size×128
        out = torch.cat((image_out, visit_out_weekdayAndWeek, visit_out_weekdayAndHour), 1)
        out = self.fc(out)

        return F.softmax(out, dim=1)
    
