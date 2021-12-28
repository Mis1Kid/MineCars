import torch
from torch import nn
import torchvision

config = {
    'MultiHeadAttention': {'numHeads': 4, 'dim': 16, 'innerDim': 4},
    'MLP': {'layers': [16, 64, 16]},
    'Vit': {'inChannels': 3, 'imgSize': (32, 32), 'patchSize': 8, 'numEncoders': 4, 'numHeads': 4, 'numLables': 10},
    'Res': {'in_channels': 3, }
}

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

vocRoot = "/home/hcl/workspace/vit-pytorch/dataset/"

cifar10Root = "/home/hcl/workspace/DataSet/"

DataSetRoot = "/home/hcl/workspace/DataSet/MineCars"

TrainInfoRoot = '/home/hcl/workspace/vit-pytorch/trainInfo/'

ModelSaveRoot = '/home/hcl/workspace/vit-pytorch/weights/'

# 预设参数
NUMCLASS = 3
BATCH_SIZE = 8
EPOCH = 500
RESNET_NUMFC=8192
SHUFFLENETV2_NUMFC=1024
IMAGE_SIZE=(1280,720)
INPUT_SHAPE=(400,400)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



LABEL_NAMES=['things','stone','mine','others']
TrainRate=5