
import os
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import  DataLoader
from dataset import UntrimmedDataset
from model import CPD_SSL


data_root = os.path.join(os.getcwd(), 'slice_data')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_size = 32
epoch=200

        # nfft, hop, batch, kernel, stride, threshold
config = [[32, 16, 32, 32, 16],
          [32, 16, 32, 64, 32],
          [32, 16, 64, 32, 16],
          [32, 16, 64, 64, 32],
          [32, 16, 128, 32, 16],
          [32, 16, 128, 64, 32],
          [64, 32, 32, 32, 16],
          [64, 32, 32, 64, 32],
          [64, 32, 64, 32, 16],
          [64, 32, 64, 64, 32],
          [64, 32, 128, 32, 16],
          [64, 32, 128, 64, 32],
          ]

def train_all(config, epochs):
    
    for c in config:
        
        n_fft = c[0]
        hop_length = c[1]
        batch_size = c[2]
        kernel_size = c[3]
        stride_size = c[4]
        
        folder_name = 'outputs_nfft' + str(n_fft) + '_h_' + str(hop_length) + '_b_' + str(batch_size) + '_k_' + str(kernel_size) + '_s_' + str(stride_size)
        
        train_dataset = UntrimmedDataset(root_dir=data_root,
                                kernel_size= kernel_size,
                                stride=stride_size,
                                device=device,
                                n_fft=n_fft,
                                hop_length=hop_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
        
        test_dataset = UntrimmedDataset(root_dir=data_root,
                                kernel_size= kernel_size,
                                stride=stride_size,
                                device=device,
                                n_fft=n_fft,
                                hop_length=hop_length)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
        
        
        transform = transforms.Compose([
            transforms.Resize((128, 128), antialias=True),  # 이미지 크기를 256x256으로 조정
        ])
        
        backbones = ['SqeezeNet','ShuffleNet','RegNet','MobileNet','EfficientNet','MnasNet']
        
        
        for backbone in backbones:
            cpd = CPD_SSL(backbone=backbone, feature_size=feature_size, device=device)
            cpd.backbone.to(device)
            cpd.train_set(folder_name, train_loader, test_loader, epochs, transform)

train_all(config,epoch)