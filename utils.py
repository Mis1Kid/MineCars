import re
import os
from functools import cmp_to_key
import torch
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
    return inta-intb

def readFromDirToList(pathRoot,key='correctRate'):
    l=[]
    dirs=[pathRoot+dir for dir in os.listdir(pathRoot) if dir.endswith('txt') or dir.endswith('pth')]
    dirs.sort(key=cmp_to_key(sortList))
    for filePath in dirs:
        info=readFromFile(filePath)
        if isinstance(info, list):
            l=l+info
        elif isinstance(info,dict):
            l=l+[info[key]]
    return l