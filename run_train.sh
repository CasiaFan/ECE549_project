#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
feat_name="resnet18"
mask_name=None
loss="l1"
dataset="BIRD"
partial_label=False
python train.py --feat_name=$feat_name \
                --mask_name=$mask_name \
                --mask_loss=$loss \
                --partial_label=$partial_label \
                --image_size=224 \
                --num_classes=8 \
                --batch_size=24 \
                --num_epochs=30 \
                --model_save_path=test/${feat_name}_${mask_name}_loss_${loss}_dataset_${dataset}_partial_${partial_label} \
                --device="cuda:0" \
                --pretrained_weights=None \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset=${dataset} \
                --num_gpus=1 \
                --dilute_mask=0 \
                --mask_weight=1.0 \
                --num_mask_blocks=4