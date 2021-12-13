#!/bin/bash
model_dir="test/resnet18_mask_rasaee_loss_CE_dataset_BUSI_partial_False"
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
               --num_blocks=4 \
               --return_mask=True \
               image2mask \
               --seg_image_list="test/draw_mask.txt" \
               --mask_save_file="test/test_mask.png"
               #accuracy
               #saliency \
               #--image_path="/shared/anastasio5/COVID19/data/Dataset_BUSI_with_GT/malignant/malignant (11).png" \
               #--saliency_file="test/test_saliency.png"
    echo "Model processed: $full_path"
    echo "======================="
done 
