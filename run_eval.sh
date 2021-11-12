#!/bin/bash
model_dir="model"
declare -a StringArray=("res50_mask.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name="resnet50_attention_mask" \
               --num_classes=3 \
               --model_weights=$full_path \
               --image_size=448 \
               --device="cpu" \
               --dataset="BUSI" \
               --multi_gpu=False \
               image2mask \
               --seg_image_list="draw_mask.txt" \
               --mask_save_file="test/test_mask.png"
    echo "Model processed: $model"
    echo "======================="
done 
