import torch
from torch import nn
import torchvision
from model.vit import Vit

config = {
    'MultiHeadAttention': {'numHeads': 4, 'dim': 16, 'innerDim': 4},
    'MLP': {'layers': [16, 64, 16]},
    'Vit': {'inChannels': 3, 'imgSize': (32, 32), 'patchSize': 8, 'numEncoders': 4, 'numHeads': 4, 'numLables': 10},
    'Res':{'in_channels':3,'imgSize':(32,32), 'numLabels':10}
}

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

vocRoot = "/home/hcl/workspace/vit-pytorch/dataset/"
cifar10Root = "/home/hcl/workspace/DataSet/"
trainInfoRoot = '/home/hcl/workspace/vit-pytorch/trainInfo/'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 预设参数
CLASS_NUM = 10
BATCH_SIZE = 16
EPOCH = 500
model_save_root = '/home/hcl/workspace/vit-pytorch/weights/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(root=cifar10Root, train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR10(root=cifar10Root, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE*BATCH_SIZE, shuffle=True, num_workers=8)
