import cv2
import torch
from tools.utils import *
from config.config import *
from dataset.dataset import MineCars
from model.denseNet import DenseNet121
import time

model = dense=DenseNet121()
testset = MineCars(path=DataSetRoot, labelpath='test', imageSize=(1280, 720),
                    train=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE * BATCH_SIZE, shuffle=True, num_workers=8)
state_dict=torch.load('weights/MineCars/checkpoint-epoch-70.pth')['net']
print(validate(model, testloader, device, state_dict))