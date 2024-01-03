import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import librosa
import numpy as np
from interpolate import interpolate_data

class UntrimmedDataset(Dataset):
    def __init__(self, root_dir, kernel_size, stride, device, n_fft, hop_length):
        self.root_dir = root_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.data = pd.DataFrame(columns=['x', 'y', 'z', 'uptime', 'class'])
        
        files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.csv')]
        for file in files:
            data_pd = pd.read_csv(file)
            
            data_length = len(data_pd)
            stride_n = (data_length - kernel_size) // stride
            adjusted_length = kernel_size + (stride_n * stride)
            data_pd = data_pd.head(adjusted_length)
            
            fileinfo_pd = pd.DataFrame({'end_file': [False] * adjusted_length})
            fileinfo_pd.iloc[0] = True
            data_pd = data_pd[['x', 'y', 'z', 'uptime', 'class']]
            data_pd = pd.concat([data_pd, fileinfo_pd], axis=1)
            
            x_tensor = torch.tensor(data_pd['x'].values, dtype=torch.float32)
            y_tensor = torch.tensor(data_pd['y'].values, dtype=torch.float32)
            z_tensor = torch.tensor(data_pd['z'].values, dtype=torch.float32)

            uptime_tensor = torch.tensor(data_pd['uptime'].values, dtype=torch.float32)
            data_tensor = torch.tensor(data_pd[['x', 'y', 'z']].values, dtype=torch.float32)

            interpolated_x = self.interpolate(uptime_tensor.numpy(),x_tensor.numpy()) 
            interpolated_y = self.interpolate(uptime_tensor.numpy(),y_tensor.numpy()) 
            interpolated_z = self.interpolate(uptime_tensor.numpy(),z_tensor.numpy()) 

            interpolated_data = pd.DataFrame({
                'x': interpolated_x,
                'y': interpolated_y,
                'z': interpolated_z,
                'uptime': uptime_tensor.numpy()  # 새로운 'uptime' 열 추가
                })
            
            self.data = pd.concat([pd.DataFrame(interpolated_data.transpose(), columns=['x', 'y', 'z']), data_pd[['uptime', 'class', 'end_file']]], axis=1)
        
        self.data_len = int((len(self.data) - kernel_size) / stride + 1)
        
    def interpolate(self,uptime,data):
        interpolated_data = interpolate_data(uptime, data,new_interval=1e-4 )
        return interpolated_data
    
    def stft(self, waveform):
        window_size = self.n_fft
        hop_size = self.hop_length

        min_length = window_size + (2 * hop_size)
        if len(waveform[0]) < min_length:
            waveform = np.pad(waveform, ((0, 0), (0, min_length - len(waveform[0]))), mode='constant')

        num_segments = (len(waveform[0]) - window_size) // hop_size + 1
        if num_segments <= 0:
            raise ValueError("Input signal is too short to compute STFT")

        segments = np.zeros((len(waveform), num_segments, window_size))

        for i in range(num_segments):
            start = i * hop_size
            end = start + window_size
            segments[:, i, :] = waveform[:, start:end]
        windowed_segments = segments * np.hanning(window_size)
        stft_result = np.fft.fft(windowed_segments, axis=-1)
        return np.abs(stft_result)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.kernel_size
        data = self.data.iloc[start_idx:end_idx]
                
        # STFT 변환
        data_xyz = torch.tensor(data[['x', 'y', 'z']].values, dtype=torch.float32)
        data_stft = self.stft(data_xyz.numpy().transpose())

        other_columns = data[['uptime', 'class', 'end_file']]

        try:
            class_tensor = torch.tensor(data['class'].mode()[0])
        except:
            print(f'idx : {idx}')
            print(f'data : {data}')
            print(f'data["class"] : {data["class"]}')
            exit()
        is_new = False
        if data['end_file'].isin([True]).any():
            is_new = True

        return torch.tensor(data_stft, dtype=torch.float32), class_tensor, is_new

if __name__ == '__main__':
    
    data_root = '/Users/green/Desktop/heekim/driverBehaviorDataset/Untrimmed_acc'
    n_fft = 2048  # STFT의 FFT 크기 설정
    hop_length = 512  # STFT의 hop 길이 설정
    
    dataset = UntrimmedDataset(root_dir=data_root,
                               kernel_size=512,
                               stride=256,
                               device='cpu',
                               n_fft=n_fft,
                               hop_length=hop_length)
    
    dataloader = DataLoader(dataset, batch_size=4)

    for i, (data, class_tensor, is_new) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print("Data shape:", data.shape)
        print("Class tensor:", class_tensor)
        print("Is new:", is_new)
        
        if i == 2:  # 처음 3개의 배치만 확인
            break