import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from resnet_2d import *

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

validimage_tensor = torch.load("./imagepreprocessdataset/trainimagetensor.pt")
trainimage_tensor = torch.load("./imagepreprocessdataset/validimagetensor.pt")
validlabel = torch.load("./imagepreprocessdataset/trainlabel.pt")
trainlabel = torch.load("./imagepreprocessdataset/validlabel.pt")

class CustomDataset(Dataset): 
  def __init__(self, x, y, transforms):
    self.x_data = x
    self.y_data = y
    self.transforms = transforms

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    x = self.transforms(x)
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

train_transform = transforms.Compose([
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
valid_transform = transforms.Compose([
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
trainlabel = trainlabel.unsqueeze(-1)
validlabel = validlabel.unsqueeze(-1)

print(trainlabel[:10])
print(trainimage_tensor[:10])