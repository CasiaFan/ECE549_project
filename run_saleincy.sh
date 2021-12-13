#!/bin/bash
model_path="/home/zongfan2/Documents/ECE549_project/ECE549_project/test/resnet50_mask_rasaee_loss_CE_dataset_BUSI_partial_False/best_model.pt"
feat_name="resnet50"
mask_name="mask_rasaee"
dataset="BUSI"
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
                        --device="cpu" \
                        --dataset=$dataset \
                        --multi_gpu=False \
                        --num_blocks=4 \
                        --return_mask=False \
                           saliency \
                        --image_path="/shared/anastasio5/COVID19/data/Dataset_BUSI_with_GT/benign/benign (400).png" \
                        --saliency_file="test/busi_saliency.png"
