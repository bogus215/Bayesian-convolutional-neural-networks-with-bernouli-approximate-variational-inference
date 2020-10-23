import torch
import torch.nn as nn
from mc_dropout import MCDropout



# lenet - all
class LeNet_all(nn.Module):
    def __init__(self, droprate=0.5):
        super(LeNet_all, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5,padding=2), # conv1
            nn.ReLU(inplace=True),# relu1
            nn.Dropout2d(p=droprate), #dropout1
            nn.MaxPool2d(2,stride=2), # maxpool1
            nn.Conv2d(20,50,kernel_size=5,padding=2), # conv2
            nn.ReLU(inplace=True), # relu2
            nn.Dropout2d(p=droprate), # dropout2
            nn.MaxPool2d(2,stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(50*7*7,500), # dense1
            nn.ReLU(inplace=True), # relu3
            nn.Dropout(p=droprate), # dropout3
            nn.Linear(500,10) # dense2
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# lenet - ip
class LeNet_ip(nn.Module):
    def __init__(self, droprate=0.5):
        super(LeNet_ip, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5,padding=2), # conv1
            nn.ReLU(inplace=True),# relu1
            nn.MaxPool2d(2,stride=2), # maxpool1
            nn.Conv2d(20,50,kernel_size=5,padding=2), # conv2
            nn.ReLU(inplace=True), # relu2
            nn.MaxPool2d(2,stride=2)) # maxpool2

        self.classifier = nn.Sequential(
            nn.Linear(50*7*7,500), # dense1
            nn.ReLU(inplace=True), # relu3
            nn.Dropout(p=droprate),
            nn.Linear(500,10) # dense2
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Bayes_lenet - all
class DropOutLenet_all(nn.Module):

    def __init__(self, conv_dropout: float, dense_dropout: float,force_dropout: bool):
        super(DropOutLenet_all,self).__init__()

        self.dense_dropout = dense_dropout
        self.force_dropout = force_dropout

        # features
        self.conv_1 = nn.Conv2d(1,20,kernel_size=5, padding=2 , bias=False) # conv1 - kernel
        self.conv_1_bias = nn.Parameter(torch.randn(20)) # conv1 - bias
        self.relu_1 = nn.ReLU(inplace=True)
        self.dropout_1 = MCDropout(conv_dropout, self.force_dropout)
        self.pool_1 = nn.MaxPool2d(2 , stride=2)
        self.conv_2 = nn.Conv2d(20,50,kernel_size=5, padding=2 , bias=False) # conv2 - kernel
        self.conv_2_bias = nn.Parameter(torch.randn(50)) # conv2 - bias
        self.relu_2 = nn.ReLU(inplace=True)
        self.dropout_2 = MCDropout(conv_dropout, self.force_dropout)
        self.pool_2 = nn.MaxPool2d(2, stride=2)

        # classifier
        self.dense_1 = nn.Linear(50*7*7,500, bias=False)
        self.dense_1_bias = nn.Parameter(torch.randn(500))
        self.relu_3 = nn.ReLU(inplace=True)
        self.dropout_3 = MCDropout(dense_dropout, self.force_dropout)
        self.dense_2 = nn.Linear(500,10, bias=False)
        self.dense_2_bias = nn.Parameter(torch.randn(10))

    def forward(self, x):

        out = self.conv_1(x) + self.conv_1_bias.view(1,20,1,1) # conv
        out = self.relu_1(out) # relu
        out = self.dropout_1(out) # drop
        out = self.pool_1(out) # pool

        out = self.conv_2(out) + self.conv_2_bias.view(1,50,1,1) # conv
        out = self.relu_2(out) # relu
        out = self.dropout_2(out) # drop
        out = self.pool_2(out) # pool

        out = torch.flatten(out, 1)

        out = self.dense_1(out) + self.dense_1_bias.view(1,500)
        out = self.relu_3(out)
        out = self.dropout_3(out)
        out = self.dense_2(out) + self.dense_2_bias.view(1,10)

        return out



# Bayes_lenet - ip
class DropOutLenet_ip(nn.Module):

    def __init__(self, conv_dropout: float, dense_dropout: float,force_dropout: bool):
        super(DropOutLenet_ip, self ).__init__()
        self.dense_dropout = dense_dropout
        self.force_dropout = force_dropout

        # features
        self.conv_1 = nn.Conv2d(1,20,kernel_size=5, padding=2 , bias=False) # conv1 - kernel
        self.conv_1_bias = nn.Parameter(torch.randn(20)) # conv1 - bias
        self.relu_1 = nn.ReLU(inplace=True)
        self.pool_1 = nn.MaxPool2d(2 , stride=2)
        self.conv_2 = nn.Conv2d(20,50,kernel_size=5, padding=2 , bias=False) # conv2 - kernel
        self.conv_2_bias = nn.Parameter(torch.randn(50)) # conv2 - bias
        self.relu_2 = nn.ReLU(inplace=True)
        self.pool_2 = nn.MaxPool2d(2, stride=2)

        # classifier
        self.dense_1 = nn.Linear(50*7*7,500, bias=False)
        self.dense_1_bias = nn.Parameter(torch.randn(500))
        self.relu_3 = nn.ReLU(inplace=True)
        self.dense_2 = nn.Linear(500,10, bias=False)
        self.dense_2_bias = nn.Parameter(torch.randn(10))
        self.dropout_1 = MCDropout(dense_dropout)

    def forward(self, x):

        out = self.conv_1(x) + self.conv_1_bias.view(1,20,1,1) # conv
        out = self.relu_1(out) # relu
        out = self.pool_1(out) # pool

        out = self.conv_2(out) + self.conv_2_bias.view(1,50,1,1) # conv
        out = self.relu_2(out) # relu
        out = self.pool_2(out) # pool

        out = torch.flatten(out, 1)

        out = self.dense_1(out) + self.dense_1_bias.view(1,500)
        out = self.relu_3(out)
        out = self.dropout_1(out)
        out = self.dense_2(out) + self.dense_2_bias.view(1,10)

        return out
