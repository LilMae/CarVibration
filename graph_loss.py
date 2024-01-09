import matplotlib.pyplot as plt
import json
import os
# JSON 파일 경로
root = os.getcwd()
#----------Experiment Info
# model = 'RegNet'
# batch = '32'
# kernel_size = '64'
# stride = '32'
#------------------------
def loss_graph(path, model, n_fft, hop_length, batch, kernel_size, stride):
    file_path =  os.path.join(root,path,model,"result.json")

    new_folder = os.path.join(root,path,model,'loss_graph')
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
    model_info = (
    f"Model: {model}\nBatch Size: {str(batch)}\nNumber of FFT: {str(n_fft)}\nHop Length: {str(hop_length)}\nKernel Size: {str(kernel_size)}\nStride: {str(stride)}")

    plt.legend(model_info, loc='upper right')
    os.chdir(new_folder)
    plt.savefig(model+'.png',bbox_inches='tight')
    plt.show()
