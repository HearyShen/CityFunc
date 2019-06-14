import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)		# conv1: in_channels=3 (R,G,B), out_channels=6 (6 kernels), kernel_size=5 (5×5 kernel), stride=1 (default)
        self.pool = nn.MaxPool2d(2, 2)      # pool: kernel_size=2 (2×2 kernel), stride=2
        self.conv2 = nn.Conv2d(6, 16, 5)    # conv2: in_channels=6 (conv1 out), out_channels=16 (16 kernels), kernel_size=5 (5×5 kernel), stride=1 (default)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fc1: in_features=160, out_features=120
        self.fc2 = nn.Linear(120, 84)          # fc2: in_features=120, out_features= 84
        self.fc3 = nn.Linear(84, 3)            # fc3: in_features= 84, out_features=  3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # x_out: (N=4, C=6, H=398, W=398)
        x = self.pool(F.relu(self.conv2(x)))    # x_out: (N=4, C=16, H=197, W=197) total 2483776
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
