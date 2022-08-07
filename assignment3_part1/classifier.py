# my random start
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class ClassifierSimple(nn.Module):
    # TODO: implement me
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=1,
                     padding=2, bias=False)
        self.conv2 = nn.Conv2d(64, 256, 3, stride=1,
                     padding=1, bias=False)
        self.conv3 = nn.Conv2d(256, 64, 3, stride=1,
                     padding=1, bias=False)

        self.pool = nn.AvgPool2d(3, 3)
#         self.pool2 = nn.MaxPool2d(3, 3,stride=2)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm2d(16)
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x)))) 
        x = self.pool(self.bn3(F.relu(self.conv3(x)))) 

        x = x.view(-1,64*x.size()[2]**2)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ClassifierVGG(nn.Module):
    # TODO: implement me
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1,
                     padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1,
                     padding=1, bias=False)
        
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1,
                     padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1,
                     padding=1, bias=False)
        
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1,
                     padding=1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1,
                     padding=1, bias=False)
        
        self.conv7 = nn.Conv2d(128, 256, 3, stride=1,
                     padding=1, bias=False)
        self.conv8 = nn.Conv2d(256, 256, 3, stride=1,
                     padding=1, bias=False)
        
        self.pool = nn.MaxPool2d(3, 3)
#         self.pool2 = nn.MaxPool2d(3, 3,stride=2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
#         self.bn4 = nn.BatchNorm2d(16)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, NUM_CLASSES)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))# 16 filters out
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))# 32 filters out
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x)) # 64 filters out
        x = self.pool(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))# 128 filters out
        x = self.pool(x)
        
#         print(x.size())

        x = x.view(-1,256*x.size()[2]**2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
#         x = self.softmax(x)
        return x


class ClassifierInception(nn.Module):
    # TODO: implement me
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1,
                     padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1,
                     padding=1, bias=False)
        
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1,
                     padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 128, 1, stride=1,
                     padding=1, bias=False)
        
        self.pool = nn.MaxPool2d(3, 3)
        self.AvgPool = nn.AvgPool2d(5, 5)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(16)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, NUM_CLASSES)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("start",x.size())
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        print("Conv1",x.size())
        x = self.pool(x)
        print("pool1",x.size())
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        print("Conv2",x.size())
        x = self.pool(x)
        print("pool2",x.size())
        
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        print("Conv3",x.size())
        x = self.pool(x)   
        print("pool3",x.size())
        
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        print("Conv4",x.size())
        x = self.AvgPool(x)
        print("pool4",x.size())
        
        x = x.view(-1,128*x.size()[2]**2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
#         x = self.softmax(x)
        return x