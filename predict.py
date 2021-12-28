import cv2
import torch
from tools.utils import *
from config.config import *
from model.dataset import MineCars
from model.resnet import ResNet
import time

rootpath='/home/hcl/workspace/DataSet/MineCars/'
labelfilename='202003160825100Fr.txt'
totalLabelPath = os.path.join(rootpath, 'yolo', labelfilename)
totalImagePath = os.path.join(rootpath, 'data2', labelfilename[:-3] + 'bmp')
print("totalImagePath is %s"%totalImagePath)
print("totalLabelPath is %s"%totalLabelPath)


image=cv2.imread(totalImagePath)
imageSize=IMAGE_SIZE
labelInfo = readYoloTxt(totalLabelPath)
assert(labelInfo is not None)
label = labelInfo[0]
cx, cy, w, h = labelInfo[1:]
x1, y1, x2, y2 = xywhTop1p2(cx, cy, w, h, imageSize)
img=image[y1:y2,x1:x2]
img=cv2.resize(img,INPUT_SHAPE)
img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

model = ResNet([3, 4, 23, 3]).to(device)
model.eval()
input=transform(img).to(device)
print("input shape is {}".format(input.shape))
timebegin=time.time()
output=model(input[None])
predictLabelIndex=torch.argmax(output).cpu().numpy()
predictLabel=LABEL_NAMES[predictLabelIndex]
timeend=time.time()
print("time pass {}".format(timeend-timebegin))
print("predict label is --{}--".format(predictLabel))

cv2.putText(img,predictLabel , (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255))
cv2.imshow('raw',image)
cv2.imshow('after',img)
cv2.waitKey()

