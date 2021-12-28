import os
import random
import cv2
from torch.utils.data import Dataset

from config.config import *
from tools.utils import readYoloTxt, xywhTop1p2


class MineCars(Dataset):  # for training/testing

    def __init__(self, path,  labelpath="labels", imageSize=(1280, 720), train=True, transform=None):
        super(MineCars, self).__init__()
        self.labels = []
        self.data = []
        self.imageNames = []
        self.transform = transform
        self.path = path
        self.train = train
        self.TrainRate = TrainRate
        self.labelFileNames = os.listdir(os.path.join(path, labelpath))
        self.imagePathes = []
        for labelfilename in self.labelFileNames:
            if labelfilename != "classes.txt":
                totalLabelPath = os.path.join(path, labelpath, labelfilename)
                totalImagePath = os.path.join(
                    path, 'data', labelfilename[:-3] + 'bmp')
                labelInfo = readYoloTxt(totalLabelPath)
                if labelInfo is not None:
                    label = labelInfo[0]
                    cx, cy, w, h = labelInfo[1:]
                    x1, y1, x2, y2 = xywhTop1p2(cx, cy, w, h, imageSize)
                    self.imagePathes.append(totalImagePath)
                    self.labels.append([label, x1, y1, x2, y2])
                    self.imageNames.append(labelfilename[:-3])
        if train == True:
            self.imageNames = [self.imageNames[i] for i in range(
                len(self.imageNames)) if i % self.TrainRate != 0]
            self.imagePathes = [self.imagePathes[i] for i in range(
                len(self.imagePathes)) if i % self.TrainRate != 0]
            self.labels = [self.labels[i] for i in range(
                len(self.labels)) if i % self.TrainRate != 0]

        else:
            self.imageNames = [self.imageNames[i] for i in range(
                len(self.imageNames)) if i % self.TrainRate == 0]
            self.imagePathes = [self.imagePathes[i] for i in range(
                len(self.imagePathes)) if i % self.TrainRate == 0]
            self.labels = [self.labels[i] for i in range(
                len(self.labels)) if i % self.TrainRate == 0]

    def __len__(self):
        return len(self.imagePathes)

    def __getitem__(self, index):
        image = cv2.imread(self.imagePathes[index])
        label, x1, y1, x2, y2 = self.labels[index]
        img = image[y1:y2, x1:x2]
        img = cv2.resize(img, INPUT_SHAPE)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        return img, label
