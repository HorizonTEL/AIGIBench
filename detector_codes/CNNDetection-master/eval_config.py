from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)


# dataroot = '/data/ziqiang/Benchmark'
# dataroot = '/data/ziqiang/jpeg'
# dataroot = '/data/ziqiang/noise'
dataroot = '/data/ziqiang/sample'


vals = ['BlendFace', 'BLIP', 'CommunityAI', 'DALLE-3', 'E4S', 'FaceSwap', 'FLUX1-dev', 'GLIDE', 'Imagen3', 'Infinite-ID', 'InstantID', 'InSwap', 'IP-Adapter', 'Midjourney', 'PhotoMaker', 'ProGAN', 'R3GAN', 'SD3', 'SDXL', 'SimSwap', 'SocialRF', 'StyleGAN-XL', 'StyleGAN3', 'StyleSwim', 'WFIR']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# model
model_path = '/data/ziqiang/yjz/code/CNNDetection-master/checkpoints/4class-car-cat-chair-horse2025_04_12_14_39_53/model_epoch_best.pth'