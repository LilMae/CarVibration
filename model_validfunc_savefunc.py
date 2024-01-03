import torch
import torchvision
import os


class CPD_SSL():
    def __init__(self, backbone, feature_size, device):
        self.backbone = self.backbone_load(backbone, feature_size, device)
        self.backbone_name = backbone
        
    def backbone_load(self, backbone, feature_size, device):
        
        # 1. SqueezeNet
        if backbone == 'Sqeezenet':
            squeeze_net = torchvision.models.squeezenet1_1(progress=True).to(device)
            squeeze_net.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
                torch.nn.Flatten(),
                torch.nn.Linear(512, feature_size, bias=True))
            return squeeze_net
            
        # 2. ShuffleNet
        elif backbone == 'ShuffleNet':
            shuffle_net = torchvision.models.shufflenet_v2_x2_0().to(device)
            shuffle_net.fc = torch.nn.Linear(in_features=2048, out_features=feature_size, bias=True)
            return shuffle_net
            
        # 3. RegNet
        elif backbone == 'RegNet':
            reg_net = torchvision.models.regnet_y_1_6gf().to(device)
            reg_net.fc = torch.nn.Linear(in_features=888, out_features=feature_size, bias=True)
            return reg_net
            
        # 4. MobileNet
        elif backbone == 'MobileNet':
            mobile_net = torchvision.models.mobilenet_v3_large().to(device)
            mobile_net.classifier = torch.nn.Sequential(
                torch.nn.Linear(960, 1280, bias=True),
                torch.nn.Linear(1250, feature_size, bias=True))
            return mobile_net

        # 5. EfficientNet
        elif backbone == 'EfficientNet':
            efficient_net = torchvision.models.efficientnet_b2().to(device)
            efficient_net.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3, inplace=True),
                torch.nn.Linear(in_features=1408, out_features=feature_size, bias=True))
            return efficient_net

        # 6. MnasNet
        elif backbone == 'MnasNet':
            mnas_net = torchvision.models.mnasnet1_3().to(device)
            mnas_net.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=1280, out_features=feature_size, bias=True))
            return mnas_net
        else:
            print(f'Error : Unspportable Backbone - {backbone}')

    def train_one_epoch(self, data_loader):
        
        for batch in data_loader:
            try:
                break
            except:
                break
            
        
    def save_model(self, epoch):
        # Check if the directory exists, and create it if not
        directory = os.path.dirname("./ckpt")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch
        }
        checkpoint_path = "./ckpt/" + self.backbone_name + "_" + epoch + ".pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # model 결과값 어떤식으로 나오는지 알려주시면 맞춰서 수정하겠습니다..!
    def valid_one_epoch(self, data_loader, threshold = 0.0):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0.0
        self.backbone.eval()
        
        true_correct = 0
        false_correct = 0
        total = 0
        
        
        setting_threshold = threshold
    
        with torch.no_grad():
            for index, batches in enumerate(data_loader):
                inputs, labels, is_new = batches
                inputs = inputs.to(device)
                predicted = self.backbone(inputs)
                
                for idx in range(len(batches)):               # 각 배치별
                    if predicted[idx] < setting_threshold:  # 해당 배치가 threshold 이하인지
                        l0 = labels[idx][0]
                        # l0 = len(set(labels[idx].unique().numpy()))
                        tf_flag = True
                        
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
                print(f'[Test] index: {index + 1} | Acc: {true_correct / total * 100:.4f}')
            
            print(f'[Test] epoch: {1} | Acc: {true_correct / total * 100:.4f}')



