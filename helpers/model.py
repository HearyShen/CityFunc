from torch import nn
from torchvision import models

def autofit(model, arch, num_classes):
    assert arch.startswith('alexnet') or arch.startswith('vgg') or arch.startswith('resnet')
    
    if arch.startswith('alexnet'):
        # replace AlexNet's classifier for customized num_classes
        model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    elif arch.startswith('vgg'):
        # replace VGGNet's classifier for customized num_classes
        model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    elif arch.startswith('resnet'):
        # replace ResNet's last Fully-connected layer for customized num_classes
        if arch.endswith('18') or arch.endswith('34'):
            expansion = models.resnet.BasicBlock.expansion
        else:
            expansion = models.resnet.Bottleneck.expansion
        model.fc = nn.Linear(512 * expansion, num_classes)
    # elif arch.startswith('squeezenet'):
    #     self.model.num_classes = num_classes
    #     # replace SqueezeNet's classifier for customized num_classes
    #     # Final convolution is initialized differently from the rest
    #     final_conv = nn.Conv2d(512, self.model.num_classes, kernel_size=1)
    #     self.model.classifier = nn.Sequential(
    #         nn.Dropout(p=0.5),
    #         final_conv,
    #         nn.ReLU(inplace=True),
    #         nn.AdaptiveAvgPool2d((1, 1))
    #     )
    # elif arch.startswith('densenet'):
    #     num_features
    #     self.model.classifier = nn.Linear(num_features, num_classes)
    # elif arch.startswith('inception'):
    #     self.model.fc = nn.Linear(2048, num_classes)
    # elif arch.startswith('googlenet'):
    #     self.model.fc = nn.Linear(1024, num_classes)
    # elif arch.startswith('shufflenet'):
    #     self.model.fc
    # elif arch.startswith('mobilenet'):
    #     self.model.fc
    # elif arch.startswith('resnext'):
    #     self.model.fc
    # else:

    return model