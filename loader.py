import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader , random_split, Subset
import torch
import numpy as np
import random

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

class c_loader():

    def __init__(self, args):
        super(c_loader, self).__init__()

        mnist_transform = transforms.Compose([transforms.ToTensor()])
        download_root = './MNIST_DATASET'

        dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
        dataset = Subset(dataset,random.sample(range(dataset.__len__()) , args.data_size))
        test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=2000, shuffle=True)
        del dataset




