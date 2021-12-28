from matplotlib import pyplot as plt
import os
from config import *
import re
from functools import cmp_to_key
from utils import *

if __name__=='__main__':
    lResNet=readFromDirToList(model_save_root+'Vit/')
    lVit=readFromDirToList(model_save_root+'ResNet/')
    x=[i for i in range(len(lResNet))]
    y=[i for i in range(len(lVit))]
    plt.figure('accuracy')
    plt.plot(x,lResNet,label='ResNet')
    plt.plot(y,lVit,label='Vit')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()