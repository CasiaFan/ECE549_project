#!/bin/bash
model_base_dir="test"
for model in $model_base_dir/*; do
    model_path="$model/best_model.pt"
    info=($(echo $model_path | tr "_" "\n"))
    feat_name=$(basename ${info[0]})
    mask_name=${info[1]}
    echo $feat_name
    if [ $mask_name = "mask" ]; then
        mask_name="${info[1]}_${info[2]}"
        dataset=${info[6]}
    else
        dataset=${info[5]}
    fi
    if [ $dataset = "BUSI" ]; then
        num_class=2
    else
        num_class=8
    fi  
    echo "$num_class"
    python eval.py --feat_name=$feat_name \
                --mask_name=$mask_name \
                --num_classes=$num_class \
                --model_weights=$model_path \
                --image_size=224 \
                --device="cuda:0" \
                --dataset=$dataset \
                --multi_gpu=False \
                accuracy
                #    image2mask \
                #    --seg_image_list="draw_mask.txt" \
                #    --mask_save_file="test/test_mask.png"
        echo "Model processed: $model_path"
        echo "======================="
done
