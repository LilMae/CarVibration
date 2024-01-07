import matplotlib.pyplot as plt
import json
import os
# JSON 파일 경로
root = os.getcwd()

#--------Experiment Info
model = 'SqueezeNet'
batch = '32'
kernel_size = '64'
stride = '32'
#------------------------

file_path =  os.path.join(root,"outputs",model,"result.json")
new_folder = os.path.join(root,"outputs",model,'loss_graph')
os.makedirs(new_folder)
            
with open(file_path, 'r') as file:
    data = json.load(file)

epochs = [entry["Epoch"] for entry in data]
loss_values = [entry["loss_epoch"] for entry in data]

plt.figure(figsize=(8, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-',markersize=1)
#plt.title('Model : {}'.format(model))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
model_info = "Model:" + model +"\nBatch Size : " + batch + "\nKernel Size : " + kernel_size + "\nStride : " + stride

plt.legend([model_info], loc='upper right')
os.chdir(new_folder)
plt.savefig(model+'.png',bbox_inches='tight')
plt.show()
