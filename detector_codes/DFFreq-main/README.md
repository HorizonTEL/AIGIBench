<div align="center">
 <br>
<h1>Dual Frequency Branch Framework with Reconstructed Sliding Windows Attention for AI-Generated Image Detection</h1>
 
[Jiazhen Yan](https://scholar.google.com/citations?user=QkURh8EAAAAJ&hl=zh-CN)<sup>1</sup>, [Ziqiang Li](https://scholar.google.com/citations?user=mj5a8WgAAAAJ&hl=zh-CN)<sup>1</sup>,  [Fan Wang](https://scholar.google.com/citations?user=zT1Ad0gAAAAJ&hl=zh-CN)<sup>1</sup>, [Ziwen He](https://scholar.google.com/citations?user=PjkDK9cAAAAJ&hl=zh-CN)<sup>1</sup>, [Zhangjie Fu](https://scholar.google.com/citations?user=fO9NmagAAAAJ&hl=zh-CN)<sup>1‡</sup>


<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>‡</sup></span>Corresponding author
  </p>
</div>

<sup>1</sup>Nanjing University of Information Science and Technology
<p align="center">
  <a href='https://github.com/HorizonTEL/DFFreq-main'>
    <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'>
  </a>
  <a href='https://arxiv.org/abs/2501.15253'>
    <img src='https://img.shields.io/badge/Arxiv-2501.15232-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2501.15253'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
</p>
</div>

## 🔥 News
* [2026-02-07]🎉🎉🎉 DFFreq is accepted by IEEE Transactions on Information Forensics & Security.

## ⏳ Quick Start
### 1. Installation
```
conda create -n DFFreq -y python=3.9
conda activate DFFreq
pip3 install torch torchvision
pip install -r requirements.txt 
```
### 2.Getting datasets
| Datasets          |    Paper                                                                                                               |    Url    |
|:------:           |:---------:                                                                                                             |:---------:|
| GANGen-Detection  | Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning (AAAI 2024)     | [Google Drive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj) |
| DiffusionForensics| DIRE for Diffusion-Generated Image Detection (ICCV 2023)                                                               | [Google Drive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf) |
| UniversalFakeDetect| Towards Universal Fake Image Detectors that Generalize Across Generative Models (CVPR 2023)                            | [Google Drive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-) |
| AIGCDetectBench   | PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection                                         | [ModelScope](https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/tree/master/AIGCDetectionBenchMark) |
| AIGIBench         | Is Artificial Intelligence Generated Image Detection a Solved Problem? (NeurIPS 2025)                                  | [Huggingface](https://huggingface.co/datasets/HorizonTEL/AIGIBench)/[Baidu Netdisk](https://pan.baidu.com/s/1XTwfXlfqkGxAwYLxXuZbfA?pwd=sm6v) |
### 3.Inference
Of course, you need to change [DetectionTests] in test.py when testing.

We present our inference results in log_test.log.
```
python test.py --model_path ./checkpoints/model_epoch_last.pth
```

## ⏳ Training
The training set uses four classes from CNN-Spot(CNN-generated images are surprisingly easy to spot...for now, CVPR 2020): car, cat, chair, and horse. [Baidu Netdisk](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)
```
python train.py --name 4class-car-cat-chair-horse --dataroot [training datasets path] --classes car,cat,chair,horse
```

## Citation 
```
@article{yan2026dual,
  title={Dual Frequency Branch Framework with Reconstructed Sliding Windows Attention for AI-Generated Image Detection},
  author={Yan, Jiazhen and Li, Ziqiang and Wang, Fan and He, Ziwen and Fu, Zhangjie},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}
```

## Contact
If you have any question about this project, please feel free to contact 247918horizon@gmail.com

