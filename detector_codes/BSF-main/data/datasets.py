import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from random import random, choice, shuffle
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os
import random
import torch
import json


class FixedMapping:
    def __init__(self):
        pass
    
    def __call__(self, tensor):
        v = tensor * 255.0
        mapped = v - torch.round((v / 256.0) * 100.0) / 100.0 * 256.0
        return mapped / 255.0

# import kornia.augmentation as K

ImageFile.LOAD_TRUNCATED_IMAGES = True


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


def dataset_folder(opt, root):
    # 是否为二分类任务
    if opt.mode == 'binary':
        # return binary_dataset(opt, root)
        return CustomDataset(opt)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


class CustomDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.image_paths = []
        self.real_fake_labels = []
        # 获取类别列表
        classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes

        # 遍历类别和真假子目录，收集数据
        for cls in classes:
            if '0_real' not in classes or '1_fake' not in classes:
                for subfolder in ['0_real', '1_fake']:
                    folder_path = os.path.join(opt.dataroot, cls, subfolder)
                    if os.path.exists(folder_path):
                        for img_name in os.listdir(folder_path):
                            img_path = os.path.join(folder_path, img_name)
                            self.image_paths.append(img_path)
                            self.real_fake_labels.append(0 if subfolder == '0_real' else 1)
            else:
                folder_path = os.path.join(opt.dataroot, cls)
                if os.path.exists(folder_path):
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)
                        self.image_paths.append(img_path)
                        self.real_fake_labels.append(0 if cls == '0_real' else 1)

        self.transform = transforms.Compose([
            transforms.RandomCrop(opt.cropSize) if opt.isTrain else transforms.CenterCrop(opt.cropSize),
            transforms.RandomHorizontalFlip() if opt.isTrain and not opt.no_flip else transforms.Lambda(lambda img: img),
            transforms.ToTensor(),
            FixedMapping(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        classify_label = self.real_fake_labels[index]

        return image, classify_label

    def __len__(self):
        return len(self.image_paths)


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
