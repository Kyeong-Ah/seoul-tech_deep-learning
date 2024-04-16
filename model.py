import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """
    def __init__(self, regularization=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        output = output.view(output.size(0), -1)
        output = self.relu3(self.fc1(output))
        output = self.relu4(self.fc2(output))
        output = self.fc3(output)

        return output


class LeNet5_Dropout(nn.Module):
    def __init__(self):
        super(LeNet5_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        output = output.view(output.size(0), -1)
        output = self.relu3(self.fc1(output))
        output = self.dropout1(output)
        output = self.relu4(self.fc2(output))
        output = self.dropout2(self(output))
        output = self.fc3(output)

        return output


class CustomMLP(nn.Module): # 파라미터 개수: 
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 57)
        self.fc2 = nn.Linear(57, 47)
        self.fc3 = nn.Linear(47, 10)

    def forward(self, img):
        img = img.view(img.size(0), -1)
        output = F.relu(self.fc1(img))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

