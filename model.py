import torch
import torchvision


class CPD_SSL():
    def __init__(self, backbone, feature_size, device):
        
        # 1. SqueezeNet
        if backbone == 'Sqeezenet':
            squeeze_net = torchvision.models.squeezenet1_1(progress=True).to(device)
            squeeze_net.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
                torch.nn.Flatten(),
                torch.nn.Linear(512, feature_size, bias=True))
            self.backbone = squeeze_net
            
        # 2. ShuffleNet
        elif backbone == 'ShuffleNet':
            shuffle_net = torchvision.models.shufflenet_v2_x2_0().to(device)
            shuffle_net.fc = torch.nn.Linear(in_features=2048, out_features=feature_size, bias=True)
            self.backbone = shuffle_net
            
        # 3. RegNet
        elif backbone == 'RegNet':
            reg_net = torchvision.models.regnet_y_1_6gf().to(device)
            reg_net.fc = torch.nn.Linear(in_features=888, out_features=feature_size, bias=True)
            self.backbone = reg_net
            
        # 4. MobileNet
        elif backbone == 'MobileNet':
            mobile_net = torchvision.models.mobilenet_v3_large().to(device)
            mobile_net.classifier = torch.nn.Sequential(
                torch.nn.Linear(960, 1280, bias=True),
                torch.nn.Linear(1250, feature_size, bias=True))
            self.backbone = mobile_net

        # 5. EfficientNet
        elif backbone == 'EfficientNet':
            efficient_net = torchvision.models.efficientnet_b2().to(device)
            efficient_net.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3, inplace=True),
                torch.nn.Linear(in_features=1408, out_features=feature_size, bias=True))
            self.backbone = efficient_net

        # 6. MnasNet
        elif backbone == 'MnasNet':
            mnas_net = torchvision.models.mnasnet1_3().to(device)
            mnas_net.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=1280, out_features=feature_size, bias=True))
            self.backbone = mnas_net
        else:
            print(f'Error : Unspportable Backbone - {backbone}')

    def train_one_epoch(self):
        
    def valid_one_epoch(self):
        
        



