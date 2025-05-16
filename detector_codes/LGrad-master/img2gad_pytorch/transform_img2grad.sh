#!/bin/bash

GANmodelpath=$(cd $(dirname $0); pwd)/
Imgrootdir=$2
Saverootdir=$3
Classes='0_real 1_fake'

# Valdatas='sdv1.4' # airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor
# Valrootdir=${Imgrootdir}/val/
# Savedir=$Saverootdir/val/

# for Valdata in $Valdatas
# do
#     for Class in $Classes
#     do
#         Imgdir=${Valdata}/${Class}
#         CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py \
#             ${Valrootdir}${Imgdir} \
#             ${Savedir}${Imgdir} \
#             ./karras2019stylegan-bedrooms-256x256_discriminator.pth \
#             1
#     done
# done


# Traindatas='sdv1.4' # horse car cat chair
# Trainrootdir=${Imgrootdir}/train/
# Savedir=$Saverootdir/train/
# for Traindata in $Traindatas
# do
#     for Class in $Classes
#     do
#         Imgdir=${Traindata}/${Class}
#         CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py \
#             ${Trainrootdir}${Imgdir} \
#             ${Savedir}${Imgdir} \
#             ./karras2019stylegan-bedrooms-256x256_discriminator.pth \
#             1
#     done
# done

Testdatas='BlendFace BLIP CommunityAI DALLE-3 E4S FaceSwap FLUX1-dev GLIDE Imagen3 Infinite-ID InstantID InSwap IP-Adapter Midjourney PhotoMaker ProGAN R3GAN SD3 SDXL SimSwap SocialRF StyleGAN-XL StyleGAN3 StyleSwim WFIR'
Testrootdir=${Imgrootdir}
Savedir=$Saverootdir

for Testdata in $Testdatas
do
    for Class in $Classes
    do
        Imgdir=${Testdata}/${Class}
        CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py \
            ${Testrootdir}${Imgdir} \
            ${Savedir}${Imgdir} \
            ./karras2019stylegan-bedrooms-256x256_discriminator.pth \
            1
    done
done