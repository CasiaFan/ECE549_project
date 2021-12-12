#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
declare -a NET=("resnet18" "resnet34" "resnet50")
declare -a MASK=("None" "mask_rasaee")
declare -a LOSS=("l1" "l2" "CE")
declare -a DS=("BIRD" "BUSI")
declare -a PL=(True False)
declare -a BLK=(3 4)
# feat_name="resnet18"
# mask_name=None
# loss="l1"
# dataset="BIRD"
# partial_label=False
for feat_name in "${NET[@]}"
do
    for mask_name in "${MASK[@]}"
    do 
        for loss in "${LOSS[@]}"
        do
            for dataset in "${DS[@]}"
            do 
                for partial_label in "${PL[@]}"
                do
                    for num_blocks in "${BLK[@]}"
                    do
                        if [ "$loss" = "l1" ];then
                            mask_weight=0.2
                        elif [ "$loss" = "l2" ]; then
                            mask_weight=0.1
                        else 
                            mask_weight=0.5
                        fi
                        if [ "$dataset" = "BUSI" ]; then
                            num_classes=2
                        else 
                            num_classes=8
                        fi
                        python train.py --feat_name=$feat_name \
                        --mask_name=$mask_name \
                        --mask_loss=$loss \
                        --partial_label=$partial_label \
                        --image_size=224 \
                        --num_classes=$num_classes \
                        --batch_size=16 \
                        --num_epochs=40 \
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
                        --num_mask_blocks=$num_blocks
                    done
                done 
            done
        done
    done
done
