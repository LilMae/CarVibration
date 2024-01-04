
import os
import torch
import torchvision
from torch.utils.data import  DataLoader
from dataset import UntrimmedDataset
from model import CPD_SSL
data_root = os.path.join(os.getcwd(), 'slice_data')
n_fft = 32
hop_length = int(n_fft/4)
device = 'cpu'
feature_size = 32

dataset = UntrimmedDataset(root_dir=data_root,
                               kernel_size= 64,
                               stride=32,
                               device='cpu',
                               n_fft=n_fft,
                               hop_length=hop_length)

dataloader = DataLoader(dataset, batch_size=32, drop_last=True)

cpd = CPD_SSL(backbone='SqeezeNet', feature_size=32, device='cpu')

from torchvision import transforms, utils
transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),  # 이미지 크기를 256x256으로 조정
])

cpd.train(dataloader, epoch=200, transforms=transform)