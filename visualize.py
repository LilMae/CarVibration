import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import torch

from interpolate import interpolate_data 


trimmed = 'Trimmed_accline'
data_root = os.path.join('/Users/green/Documents/GitHub/CarVibration',trimmed)
#-----------------------------
n_fft = 32
hop_length = int(n_fft/4)
device = 'cpu'
feature_size = 32
kernel_size = 512
stride = 128
idx = 0 
#-----------------------------

info_class = ['/Aggressive_acceleration',
              '/Aggressive_breaking',
              '/Aggressive_left_lane_change',
              '/Aggressive_left_turn',
              '/Aggressive_right_lane_change',
              '/Aggressive_right_turn',
              '/Non_aggressive_event']

#데이터 선택
data_pd = pd.read_csv(data_root + info_class[-1]+'/1.csv')
data_length = len(data_pd)
stride_n = (data_length - kernel_size) // stride


data_pd = data_pd[['x', 'y', 'z', 'uptime']]

x_tensor = torch.tensor(data_pd['x'].values, dtype=torch.float32)
y_tensor = torch.tensor(data_pd['y'].values, dtype=torch.float32)
z_tensor = torch.tensor(data_pd['z'].values, dtype=torch.float32)

uptime_tensor = torch.tensor(data_pd['uptime'].values, dtype=torch.float32)
data_tensor = torch.tensor(data_pd[['x', 'y', 'z']].values, dtype=torch.float32)

interpolated_x = interpolate_data(uptime_tensor.numpy(),x_tensor.numpy()) 
interpolated_y = interpolate_data(uptime_tensor.numpy(),y_tensor.numpy()) 
interpolated_z = interpolate_data(uptime_tensor.numpy(),z_tensor.numpy()) 

interpolated_data = pd.DataFrame({
    'x': interpolated_x,
    'y': interpolated_y,
    'z': interpolated_z,
    'uptime': uptime_tensor.numpy()  
    })

data = pd.concat([pd.DataFrame(interpolated_data, columns=['x', 'y', 'z']), data_pd[['uptime']]], axis=1)

start_idx = idx * stride
end_idx = start_idx + kernel_size
x_data = torch.tensor(data['x'][start_idx:end_idx].values).view(-1)
y_data = torch.tensor(data['y'][start_idx:end_idx].values).view(-1)
z_data = torch.tensor(data['z'][start_idx:end_idx].values).view(-1)

x_stft = torch.stft(input=torch.tensor(x_data),n_fft=n_fft,hop_length=hop_length,return_complex=True)
y_stft = torch.stft(input=torch.tensor(y_data),n_fft=n_fft,hop_length=hop_length,return_complex=True)
z_stft = torch.stft(input=torch.tensor(z_data),n_fft=n_fft,hop_length=hop_length,return_complex=True)

data_stft = torch.stack((x_stft,y_stft,z_stft))
abs_stft = abs(data_stft)
print(abs_stft.shape)

tensor_rgb = abs_stft.permute(1, 2, 0)

plt.figure(figsize=(6, 6))
plt.imshow(tensor_rgb)
plt.axis('off')
plt.show()


