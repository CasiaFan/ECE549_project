#!/bin/bash
model_path="/home/zongfan2/Documents/ECE549_project/ECE549_project/test/resnet50_mask_rasaee_loss_CE_dataset_BIRD_partial_False/best_model.pt"
feat_name="resnet50"
mask_name="mask_rasaee"
dataset="BIRD"
if [ "$dataset" = "BUSI" ]; then
    num_class=2
else
    num_class=8
fi
python eval.py --feat_name=$feat_name \
                        --mask_name=$mask_name \
                        --num_classes=$num_class \
                        --model_weights=$model_path \
                        --image_size=224 \
                        --device="cuda:0" \
                        --dataset=$dataset \
                        --multi_gpu=False \
                        --num_blocks=4 \
                        --return_mask=True \
                           image2mask \
                        --seg_image_list="test/draw_bird.txt" \
                        --mask_save_file="test/bird_mask.png"
