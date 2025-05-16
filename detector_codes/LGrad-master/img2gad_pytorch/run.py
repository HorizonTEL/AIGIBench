import os
import subprocess
import torch

# 设置类和模型路径
classes = ['0_real', '1_fake']
gan_model_path = 'E:/learning/Deepfake/LGrad-master/img2gad_pytorch'
img_root_dir = 'E:/learning/Deepfake/dataset/CNN_data'
save_root_dir = 'E:/learning/Deepfake/dataset/img2gad_pytorch/Grad_dataset'

# 激活conda环境
subprocess.run(['conda', 'activate', 'pytorch'], shell=True)

# 初始化CUDA上下文
if torch.cuda.is_available():
    torch.tensor(1.0, device="cuda")


# 定义处理函数
def process_data(data_list, data_root_dir, save_dir):
    for data in data_list:
        for cls in classes:
            img_dir = os.path.join(data, cls)
            img_dir_path = os.path.join(data_root_dir, img_dir)
            save_dir_path = os.path.join(save_dir, img_dir + '_grad')
            command = (
                f"conda activate pytorch && "
                f"python {os.path.join(gan_model_path, 'gen_imggrad.py')} "
                f"{img_dir_path} {save_dir_path} "
                f"./karras2019stylegan-bedrooms-256x256_discriminator.pth 1"
            )
            subprocess.run(command, shell=True, check=True)


# 设置验证数据
val_datas = ['horse', 'car', 'cat', 'chair']
val_root_dir = os.path.join(img_root_dir, 'val')
save_dir = os.path.join(save_root_dir, 'val')

# 处理验证数据
process_data(val_datas, val_root_dir, save_dir)

# 设置训练数据
train_datas = ['horse', 'car', 'cat', 'chair']
train_root_dir = os.path.join(img_root_dir, 'train')
save_dir = os.path.join(save_root_dir, 'train')

# 处理训练数据
process_data(train_datas, train_root_dir, save_dir)

# 设置测试数据
test_datas = [
    'biggan', 'deepfake', 'gaugan', 'stargan',
    'cyclegan/apple', 'cyclegan/horse', 'cyclegan/orange',
    'cyclegan/summer', 'cyclegan/winter', 'cyclegan/zebra',
    'progan/airplane', 'progan/bicycle', 'progan/bird', 'progan/boat',
    'progan/bottle', 'progan/bus', 'progan/car', 'progan/cat', 'progan/chair',
    'progan/cow', 'progan/diningtable', 'progan/dog', 'progan/horse',
    'progan/motorbike', 'progan/person', 'progan/pottedplant', 'progan/sheep',
    'progan/sofa', 'progan/train', 'progan/tvmonitor', 'stylegan/bedroom',
    'stylegan/car', 'stylegan/cat', 'stylegan2/car', 'stylegan2/cat',
    'stylegan2/church', 'stylegan2/horse'
]

test_root_dir = os.path.join(img_root_dir, 'test')
save_dir = os.path.join(save_root_dir, 'test')

# 处理测试数据
process_data(test_datas, test_root_dir, save_dir)
