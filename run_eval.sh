#!/bin/bash
model_dir=""
declare -a StringArray=("best_model.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --feat_name="resnet18" \
               --mask_name="mask_rasaee" \
               --num_classes=2 \
               --model_weights=$full_path \
               --image_size=224 \
               --device="cuda:0" \
               --dataset="BUSI" \
               --multi_gpu=False \
               --acc
            #    image2mask \
            #    --seg_image_list="draw_mask.txt" \
            #    --mask_save_file="test/test_mask.png"
    echo "Model processed: $model"
    echo "======================="
done 
