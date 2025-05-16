MODEL="AIDE"
RESUME_PATH="./4class-checkpoints"

eval_datasets=(
    # "/data/ziqiang/Benchmark" \
    # "/data/ziqiang/jpeg" \
    # "/data/ziqiang/noise" \
    "/data/ziqiang/sample" \
)
for eval_dataset in "${eval_datasets[@]}"
do
    python  main_finetune.py \
        --model $MODEL \
        --data_path /data/ziqiang/yjz/dataset/ForenSynths/train \
        --eval_data_path $eval_dataset \
        --batch_size 64 \
        --output_dir $RESUME_PATH \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True
done

