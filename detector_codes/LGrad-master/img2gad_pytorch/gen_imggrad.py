
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from glob import glob
import cv2
import sys
import os
import PIL.Image
import torch.nn.functional as F
from torchvision import transforms
from models import build_model


processimg = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
#            transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
        ])


def read_batchimg(imgpath_list):
    img_list = []
    for imgpath in imgpath_list:
        # torch.unsqueeze(..., 0) 在第0维度上增加一个维度，使图像张量从[C, H, W] 变为[1, C, H, W]
        img_list.append(torch.unsqueeze(processimg(PIL.Image.open(imgpath).convert('RGB')), 0))
    # 在第0维度上将所有图像张量拼接起来，形成一个批处理张量，形状为 [N, C, H, W]，其中 N 是图像数量
    return torch.cat(img_list, 0)


def normlize_np(img):
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    return img * 255.                


# 获取图片地址
def get_imglist(path):
    ext = [".jpg", ".bmp", ".png", ".jpeg", ".webp", ".JPEG", ".PNG"]   # Add image formats here
    files = []
    # [files.extend(glob(os.path.join(path, f'*{e}'))) for e in ext]
    for e in ext:
        pattern = os.path.join(path, f'*{e}')
        matched_files = glob(pattern)
        # Convert backslashes to forward slashes
        matched_files = [f.replace('\\', '/') for f in matched_files]
        files.extend(matched_files)
    return sorted(files)

# 生成梯度图片
def generate_images():
    # 运行参数
    imgdir = sys.argv[1]
    outdir = sys.argv[2]
    modelpath = sys.argv[3]
    batch_size = int(sys.argv[4]) if len(sys.argv)>4 else 1
    print(f'Transform {imgdir} to {outdir}')
    os.makedirs(outdir, exist_ok=True)
    # 预训练模型
    model = build_model(gan_type='stylegan',
        module='discriminator',
        resolution=256,
        label_size=0,
        # minibatch_std_group_size = 1,
        image_channels=3)
    model.load_state_dict(torch.load(modelpath), strict=True)

    model.cuda()
    model.eval()

    imgnames_list = get_imglist(imgdir)
    if len(imgnames_list) == 0:
        exit()

    num_items = len(imgnames_list)
    print(f'From {imgdir} read {num_items} Img')
    minibatch_size = int(batch_size)
    numnow = 0

    for mb_begin in range(0, num_items, minibatch_size):
        # 这是minibatch_size个图片的切片组
        imgname_list = imgnames_list[mb_begin: min(mb_begin + minibatch_size, num_items)]
        # 将图片组变成tensor的[N, C, H, W]类型
        imgs_np      = read_batchimg(imgname_list)
        tmpminibatch = len(imgname_list)
        img_cuda = imgs_np.cuda().to(torch.float32)
        img_cuda.requires_grad = True
        pre = model(img_cuda)
        model.zero_grad()
        # 计算输入图像的梯度
        # pre.sum()是求和，也就是pre的标量值，就是输出值，img_cuda是输入值，(就是y是ore.sum(), x是img_cuda,dydx)
        # create_graph=True：创建一个计算图，以便进一步计算高阶梯度
        # retain_graph=True：保留计算图，以便在同一个批次中多次使用
        # allow_unused=False：如果某些变量没有被使用，抛出错误
        grad = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
        for idx in range(tmpminibatch):
            numnow += 1
            # permute(1, 2, 0) 将张量的维度从 (C, H, W) 转换为 (H, W, C)
            # .detach() 用于分离计算图，以防止对其进行进一步的操作会影响梯度计算和反向传播
            img = normlize_np(grad[idx].permute(1,2,0).cpu().detach().numpy())
            print(f'Gen grad to {os.path.join(outdir, imgname_list[idx].split("/")[-1])}, bs:{minibatch_size} {numnow}/{num_items}',end='\r')
            cv2.imwrite(os.path.join(outdir, imgname_list[idx].split('/')[-1].split('.')[0]+'.png'),img[...,::-1])
    print()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
