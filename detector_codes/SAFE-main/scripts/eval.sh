MODEL="SAFE"
RESUME_PATH="./checkpoints"


eval_datasets=(
    # "/data/ziqiang/Benchmark" \
    # "/data/ziqiang/jpeg" \
    # "/data/ziqiang/noise" \
    "/data/ziqiang/sample" \
)
for eval_dataset in "${eval_datasets[@]}"
do
    python  main_finetune.py \
        --input_size 256 \
        --transform_mode 'crop' \
        --model $MODEL \
        --eval_data_path $eval_dataset \
        --batch_size 256 \
        --num_workers 16 \
        --output_dir $RESUME_PATH \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True
done
