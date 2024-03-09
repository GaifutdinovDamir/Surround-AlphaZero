import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import os


device = ("cuda")

class Linear_QNet(nn.Module):
    def __init__(self, inputsize, outputsize):
        super().__init__()
        self.linear1 = nn.Linear(inputsize, 10)
        self.linear2 = nn.Linear(10, 4)
        #self.linear3 = nn.Linear(40, 4)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        x = self.linear2(x)
        return x

    def save(self, file_name = 'model.pth'):
        model_folder_path = ".\surround\model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 8, kernel_size=(5, 5), padding = 2)
        #self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=(5, 5), padding = 2)
        self.linear1 = nn.Linear(16*400, 4)
        #self.linear2 = nn.Linear(10, 4)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if len(x.size()) == 3:
            x = torch.flatten(x)
        else:
            x = torch.flatten(x, 1)
        #x = F.relu(self.linear1(x))
        x = self.linear1(x)
        return x
    
    def save(self, file_name = 'convmodel.pth'):
        model_folder_path = ".\surround\model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Qtrainer():
    def __init__(self, model, lr):
        self.lr = lr
        self.gamma = 0.9
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, action, state, reward, newstate, end, long = False):
        if long:
            state = torch.stack(state).to(device)
            newstate = torch.stack(newstate).to(device)
            action = torch.stack(action).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        if not long:
            #action = torch.tensor(action, dtype=torch.long)
            state = torch.unsqueeze(state, 0).to(device)
            action = torch.unsqueeze(action, 0).to(device)
            newstate = torch.unsqueeze(newstate, 0).to(device)
            reward = torch.unsqueeze(reward, 0).to(device)
            end = (end, )
        
        pred = self.model(state)
        target = pred.clone().to(device)

        for idx in range(len(end)):
            Q_new = reward[idx]
            if not end[idx]:
                Q_new += self.gamma * torch.max(self.model(newstate[idx]))
            act = torch.argmax(action).item()
            target[idx][act] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()


    
