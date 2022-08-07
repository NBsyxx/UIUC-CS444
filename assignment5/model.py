import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HEIGHT, WIDTH, lstm_seq_length

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)


# class DQN_LSTM(nn.Module):
#     def __init__(self, action_size):
#         super(DQN_LSTM, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.fc = nn.Linear(3136, 512)
#         self.head = nn.Linear(256, action_size)
#         self.fc1 = nn.Linear(512,256)
#         # Define an LSTM layer
#         self.lstm = nn.LSTM(512,256)

#     def forward(self, x, hidden = None):
#         # You might want to reshape x during train loop (and not while collection experience replay)
#         # print("before forward",x.shape)
#         batchsize = x.shape[0]
#         x = x.view(-1,1,84,84)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.fc(x.view(x.size(0), -1)))
#         # Pass the state through an LSTM

#         # if x.shape[0] != 32*lstm_seq_length:
#         #     x = F.relu(self.fc1(x))
#         #     return self.head(x), None
#         # else:
#         # print("before lstm",x.shape)
#         x = x.view(batchsize,-1,512)

#         ### CODE ###

#         lstm_output, hidden = self.lstm(x)

#         return self.head(lstm_output)[:,-1,:], hidden


# current running model on the left side
# class DQN_LSTM(nn.Module):
#     def __init__(self, action_size):
#         super(DQN_LSTM, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.fc = nn.Linear(3136, 512)
#         self.head = nn.Linear(256, action_size)
#         self.fc1 = nn.Linear(512,256)
#         # Define an LSTM layer
#         self.lstm = nn.LSTM(512,256)

#     def forward(self, x, hidden = None):
#         # You might want to reshape x during train loop (and not while collection experience replay)
#         # print("before forward",x.shape)
#         batchsize = x.shape[0]
# #         print("batchsize",batchsize)
#         x = x.view(-1,1,84,84)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.fc(x.view(x.size(0), -1)))
#         # Pass the state through an LSTM

#         # if x.shape[0] != 32*lstm_seq_length:
#         #     x = F.relu(self.fc1(x))
#         #     return self.head(x), None
#         # else:
#         # print("before lstm",x.shape)
#         x = x.view(batchsize,-1,512)

#         ### CODE ###
# #         print("lstm input",x.shape)
#         lstm_output, hidden = self.lstm(x)
# #         print("lstm output",lstm_output.shape)
#         lstm_output_reshaped = lstm_output[-1:,-1,:]
# #         print("lstm rehshaped",lstm_output_reshaped.shape)
#         return self.head(lstm_output_reshaped), hidden


# current running model on the right side
# current running model on the right side
class DQN_LSTM(nn.Module):
    def __init__(self, action_size):
        super(DQN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(256, action_size)
        self.fc1 = nn.Linear(512,256)
        # Define an LSTM layer
        self.lstm = nn.LSTM(512,256)
        print("init dqn lstm")

    def forward(self, x, hidden = None):
        # You might want to reshape x during train loop (and not while collection experience replay)
        # print("before forward",x.shape)
        batchsize = x.shape[0]
#         print("batchsize",batchsize)
        x = x.view(-1,1,84,84)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        # Pass the state through an LSTM

        # if x.shape[0] != 32*lstm_seq_length:
        #     x = F.relu(self.fc1(x))
        #     return self.head(x), None
        # else:
        # print("before lstm",x.shape)
        x = x.view(batchsize,-1,512)

        ### CODE ###
        # print("lstm input",x.shape)
        lstm_output, hidden = self.lstm(x)
        # print("lstm output",lstm_output.shape)
        lstm_output_reshaped = lstm_output[:,-1,:]
        # print("lstm rehshaped",lstm_output_reshaped.shape)
        return self.head(lstm_output_reshaped), hidden

class DQN_LSTM_Headless(nn.Module):
    def __init__(self, action_size):
        super(DQN_LSTM_Headless, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 1024)
        
        self.head = nn.Linear(256, action_size)
        self.fc1 = nn.Linear(1024,512)
        # Define an LSTM layer
        self.lstm = nn.LSTM(512,256)

    def forward(self, x, hidden = None):
        # You might want to reshape x during train loop (and not while collection experience replay)
        # print("before forward",x.shape)
        batchsize = x.shape[0]
#         print("batchsize",batchsize)
        x = x.view(-1,1,84,84)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        x = F.relu(self.fc1(x))
        # Pass the state through an LSTM
#         x = F.relu(self.fc1(x))

        # if x.shape[0] != 32*lstm_seq_length:
        #     x = F.relu(self.fc1(x))
        #     return self.head(x), None
        # else:
        # print("before lstm",x.shape)
        x = x.view(batchsize,-1,512)

        ### CODE ###
#         print("lstm input",x.shape)
        lstm_output, hidden = self.lstm(x)
#         print("lstm output",lstm_output.shape)
        lstm_output_reshaped = lstm_output[:,-1,:]
        output = F.relu(self.head(lstm_output_reshaped))
#         print("lstm rehshaped",lstm_output_reshaped.shape)
        return output, hidden