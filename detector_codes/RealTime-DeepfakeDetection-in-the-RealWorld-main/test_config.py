# root to the testsets
# Testdataroot = '/data/ziqiang/yjz/dataset/ForenSynths/test'
# Testdataroot = '/data/ziqiang/yjz/dataset/DiffusionForensics'
# Testdataroot = '/data/ziqiang/yjz/dataset/UniversalFakeDetect'
# Testdataroot = '/data/ziqiang/yjz/dataset/Genimage'
Testdataroot = '/data/ziqiang/Benchmark'
# Testdataroot = '/data/ziqiang/jpeg'
# Testdataroot = '/data/ziqiang/noise'
# Testdataroot = '/data/ziqiang/sample'

# list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# vals = ['adm', 'ddpm', 'iddpm', 'ldm', 'pndm', 'sdv1_new', 'sdv2', 'vqdiffusion']
# multiclass = [1, 1, 1, 1, 1, 1, 1, 1]

# vals = ['dalle', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg']
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0]

# vals = ['ADM', 'DALLE2', 'Glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0]

vals = ['BlendFace', 'BLIP', 'CommunityAI', 'DALLE-3', 'E4S', 'FaceSwap', 'FLUX1-dev', 'GLIDE', 'Imagen3', 'Infinite-ID', 'InstantID', 'InSwap', 'IP-Adapter', 'Midjourney', 'PhotoMaker', 'ProGAN', 'R3GAN', 'SD3', 'SDXL', 'SimSwap', 'SocialRF', 'StyleGAN-XL', 'StyleGAN3', 'StyleSwim', 'WFIR']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# model
model_path = '/data/ziqiang/yjz/code/RealTime-DeepfakeDetection-in-the-RealWorld-main/checkpoints/4class-car-cat-chair-horse2025_04_10_11_12_56/model_epoch_best.pth'