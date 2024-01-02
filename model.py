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

        # 미완성
    def valid_one_epoch(self):
        data_root = ""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        global best_acc
        self.backbone.eval() 
        
        true_correct = 0
        false_correct = 0
        total = 0
        
        hop_length = 0
        n_fft = 0
        
        batch_size = 0
        setting_threshold = 0.0
        
        dataset = UntrimmedDataset(root_dir=data_root,
                               kernel_size=512,
                               stride=256,
                               device=device,
                               n_fft=n_fft,
                               hop_length=hop_length)
    
        dataloader = DataLoader(dataset, batch_size=4)
        with torch.no_grad():
            for index, batches in enumerate(dataloader):
                inputs, labels, is_new = batches
                inputs = inputs.to(device)
                predicted = self.backbone(inputs)
                
                for idx in range(batch_size):               # 각 배치별
                    if predicted[idx] < setting_threshold:  # 해당 배치가 threshold 이하인지
                        tf_flag = True
                        
                        l0 = labels[idx][0]
                        # l0 = len(set(labels[idx].unique().numpy()))
                        
                        for l in labels[idx]:               # 실제 CP인지 확인
                            if l0 != l: break
                            else:
                                tf_flag = False
                                break
                        """
                        if l0 > 1:
                            true_correct += 1
                        else:
                            false_correct += 1
                        """
                        
                        if tf_flag:
                            true_correct += 1
                        else:
                            false_correct += 1
                
                total += predicted.size(0)
                # correct += (predicted == targets).sum().item()
            
            print(f'[Test] epoch: {1} | Acc: {true_correct / total * 100:.4f}')



