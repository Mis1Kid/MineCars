import math
import os
import re
from functools import cmp_to_key
import cv2
import torch
import torch.nn as nn
from copy import deepcopy

def readYoloTxt(filePath):
    if filePath.endswith('txt') and filePath != "classes.txt":
        with open(filePath, 'r') as f:
            l = f.read()
            l = l.replace('\n', '').split(' ')
            if len(l) > 4:
                l = [float(x) for x in l]
            else:
                return None
    return l


def readFromFile(filePath):
    if filePath.endswith('txt'):
        with open(filePath, 'r') as f:
            l = f.read()
            l = l[1:-1].replace('\n', '').split(', ')
            l = [float(x) for x in l]
            f.close()
        return l
    elif filePath.endswith('pth'):
        return torch.load(filePath)


def sortList(a, b):
    inta = int(re.search('epoch-(.+?)\.', a).group(1))
    intb = int(re.search('epoch-(.+?)\.', b).group(1))
    return inta - intb


def readFromDirToList(pathRoot, key='correctRate'):
    l = []
    dirs = [pathRoot + dir for dir in os.listdir(
        pathRoot) if dir.endswith('txt') or dir.endswith('pth')]
    dirs.sort(key=cmp_to_key(sortList))
    for filePath in dirs:
        info = readFromFile(filePath)
        if isinstance(info, list):
            l = l + info
        elif isinstance(info, dict):
            l = l + [info[key]]
    return l


def calFinalSize(imgSize, timesceil, timesfloor):
    imgsize = list(imgSize)
    for i in range(timesceil):
        imgsize[0] = math.ceil(imgsize[0] / 2)
        imgsize[1] = math.ceil(imgsize[1] / 2)
    for i in range(timesfloor):
        imgsize[0] = math.floor(imgsize[0] / 2)
        imgsize[1] = math.floor(imgsize[1] / 2)
    return imgsize[0] * imgsize[1]


def xywhTop1p2(x, y, w, h, imageSize):
    x1 = int(x * imageSize[0]) - int(w * imageSize[0] / 2)
    y1 = int(y * imageSize[1]) - int(h * imageSize[1] / 2)
    x2 = int(x * imageSize[0]) + int(w * imageSize[0] / 2)
    y2 = int(y * imageSize[1]) + int(h * imageSize[1] / 2)
    return x1, y1, x2, y2


def validate(model, testloader, device, state_dict):
    lossFn = nn.CrossEntropyLoss()
    loss = torch.zeros(1).to(device)
    correct = torch.zeros(1).to(device)
    total = 0
    modelEval = deepcopy(model)
    modelEval.load_state_dict(state_dict)
    modelEval.to(device)
    modelEval.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(testloader):
            data = data.to(device)
            labels = labels.to(device)
            output = modelEval(data)
            loss += lossFn(output, labels)
            preditLabel = torch.argmax(output, dim=-1)
            correct += torch.eq(preditLabel, labels).sum()
            total += labels.shape[0]
    return loss.item() / (i+1), correct.item() / total
