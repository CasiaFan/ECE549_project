#!/bin/bash
model_base_dir="test"
for model in $model_base_dir/*; do
    if [ -d $model ]; then
        model_path="$model/best_model.pt"
        if [ -f $model_path ]; then
            info=($(echo $model_path | tr "_" "\n"))
            feat_name=$(basename ${info[0]})
            mask_name=${info[1]}
            if [ $mask_name = "mask" ]; then
                mask_name="${info[1]}_${info[2]}"
                dataset=${info[6]}
                return_mask=True
            else
                dataset=${info[5]}
                return_mask=False
            fi
            if [ $dataset = "BUSI" ];then
                num_class=2
            else
                num_class=8
            fi 
            echo $feat_name 
            echo $mask_name
            echo $num_class
            echo $dataset 
            python eval.py --feat_name=$feat_name \
                        --mask_name=$mask_name \
                        --num_classes=$num_class \
                        --model_weights=$model_path \
                        --image_size=224 \
                        --device="cuda:0" \
                        --dataset=$dataset \
                        --multi_gpu=False \
                        --num_blocks=4 \
                        --return_mask=$return_mask \
                        accuracy
                        #    image2mask \
                        #    --seg_image_list="draw_mask.txt" \
                        #    --mask_save_file="test/test_mask.png"
                echo "Model processed: $model_path"
                echo "======================="
        fi
    fi
done
