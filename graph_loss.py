import matplotlib.pyplot as plt
import json
import os
# JSON 파일 경로
root = os.getcwd()

#--------Experiment Info
model = 'RegNet'
batch = '32'
kernel_size = '64'
stride = '32'
#------------------------

file_path =  os.path.join(root,"outputs",model,"result.json")

# JSON 파일 불러오기
with open(file_path, 'r') as file:
    data = json.load(file)

# 각각의 데이터에서 loss_epoch 값 추출
epochs = [entry["Epoch"] for entry in data]
loss_values = [entry["loss_epoch"] for entry in data]

# 시각화
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-',markersize=1)
#plt.title('Model : {}'.format(model))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
# 범례에 모델 정보 추가
model_info = "Model:" + model +"\nBatch Size : " + batch + "\nKernel Size : " + kernel_size + "\nStride : " + stride

# 범례에 텍스트로 모델 정보 추가
plt.legend([model_info], loc='upper right')
plt.savefig(model+'.png',bbox_inches='tight')
plt.show()
