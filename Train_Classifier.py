
import os
import torch
import torchvision
from torch.utils.data import  DataLoader
from dataset import UntrimmedDataset, TrimmedDataset
from model import CPD_SSL


data_root = os.path.join(os.getcwd(), 'Trimmed_accline')
model_name='RegNet'
n_fft = 32
hop_length = int(n_fft/4)
device = 'cpu'
feature_size = 32
batch_size = 1

train_dataset = TrimmedDataset(root_dir=data_root,
                               kernel_size= 64,
                               stride=16,
                               device=device,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               isTrain=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

cpd = CPD_SSL(backbone=model_name, feature_size=feature_size, device=device)
cpd.load_model()
from torchvision import transforms, utils
transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),  # 이미지 크기를 256x256으로 조정
])

cpd.train_classifier(data_loader=train_dataloader, transforms=transform)

test_dataset = TrimmedDataset(root_dir=data_root,
                               kernel_size= 64,
                               stride=16,
                               device=device,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               isTrain=True)

test_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

accuracy, precision = cpd.test_classifier(data_loader=test_dataloader, transforms=transform)
with open(f'result_{model_name}.txt','w') as f:
    f.write(f'accuracy : {accuracy}')
    f.write(f'precision : {precision}')