## Setup
There is a corresponding requirements.txt under each code, so you only need to use the following command after installing pytorch and other environments
```
pip install -r requirements.txt 
```
## Train & Test
### a. CNNDetection-master
Of course, you need to change [dataroot] and [model_path] in eval_config.py when testing
```
train: nohup python train.py --name 5class-car-cat-chair-horse-sdv1.4 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot [training datasets path]  --classes car,cat,chair,horse,sdv1.4 > nohup.out 2>&1 &
test:  python eval.py
```

### b. AIDE-main
Of course, you need to change [RESUME_PATH] and [eval_datasets] in eval_config.py when testing
```
train: nohup python main_finetune.py --model AIDE --batch_size 32 --blr 1e-4 --epochs 25 --data_path [training datasets path] --eval_data_path [evaling datasets path] > nohup.out 2>&1 &
test:  bash scripts/eval.sh
```

### c. SAFE-main
Of course, you need to change [RESUME_PATH] and [eval_datasets] in scripts/eval.sh when testing
```
train: nohup python main_finetune.py --model SAFE --data_path [training datasets path] --eval_data_path [evaling datasets path] --epochs 20 > nohup.out 2>&1 &
test:  bash scripts/eval.sh
```

### d. Others
All other methods work the same way. Of course, you need to change [DetectionTests] in test.py when testing
```
train: nohup python train.py --name 5class-car-cat-chair-horse-sdv1.4 --dataroot [training datasets path] --classes car,cat,chair,horse,sdv1.4 > nohup.out 2>&1 &
test:  python test.py --model_path [training checkpoints path]
```
