import os
import glob
import random
from sklearn.model_selection import train_test_split

def filter_bird_dataset(image_dir, seg_dir, save_dir, 
                        save_name_prefix, 
                        max_count=100,
                        select_class_name=None, 
                        select_class_count=None, 
                        test_ratio=0.2,
                        seg_image_ratio=0.25,
                        ):
    os.makedirs(save_dir, exist_ok=True)
    class_dirs = os.listdir(image_dir)
    train_file = os.path.join(save_dir, save_name_prefix+"_train.txt")
    test_file = os.path.join(save_dir, save_name_prefix+"_test.txt")
    # random select n classes
    if (select_class_name is None) and (select_class_count is not None) and isinstance(select_class_count, int):
        random.shuffle(class_dirs)
        select_class_name = class_dirs[:select_class_count]
    images = []
    for c in select_class_name:
        c_images = glob.glob(os.path.join(image_dir, c, "*.jpg"), recursive=True)
        if len(c_images) > max_count:
            random.shuffle(c_images)
            c_images = c_images[:max_count]
        images += c_images
    train_images, test_images = train_test_split(images, test_size=test_ratio, shuffle=True)
    # select 25% images to extract segmentation data
    train_img_count = len(train_images)
    test_img_count = len(test_images)
    train_seg_idx = random.sample(list(range(train_img_count)), int(train_img_count * seg_image_ratio))
    test_seg_idx = random.sample(list(range(test_img_count)), int(test_img_count * seg_image_ratio))
    train_segs = ["none" for _ in range(train_img_count)]
    test_segs = ["none" for _ in range(test_img_count)]
    for idx in train_seg_idx:
        image_name = os.path.basename(train_images[idx])
        image_class = os.path.dirname(train_images[idx])
        seg_name = os.path.join(seg_dir, image_class, image_name.replace("jpg", "png"))
        train_segs[idx] = seg_name
    for idx in test_seg_idx:
        image_name = os.path.basename(test_images[idx])
        image_class = os.path.dirname(test_images[idx])
        seg_name = os.path.join(seg_dir, image_class, image_name.replace("jpg", "png"))
        test_segs[idx] = seg_name 
    print("=======write=========")
    print("Num of train images: {}".format(train_img_count))
    print("Num of test images: {}".format(test_img_count))
    with open(train_file, "w") as f:
        for img, seg in zip(train_images, train_segs):
            f.write(img + "," + seg + "\n")
    f.close()
    with open(test_file, "w") as f:
        for img, seg in zip(test_images, test_segs):
            f.write(img + "," + seg + "\n")
    f.close() 


if __name__ == "__main__":
    image_dir="/Users/zongfan/Downloads/ECE549_project/CUB_200_2011/CUB_200_2011/images" 
    seg_dir="/Users/zongfan/Downloads/ECE549_project/bird_seg"
    save_dir="/Users/zongfan/Downloads/ECE549/ECE549_project/code/data"
    save_name_prefix="bird"
    max_count=100
    select_class_name=None
    select_class_count=8
    test_ratio=0.2
    seg_image_ratio=0.25
    filter_bird_dataset(image_dir, seg_dir, save_dir, save_name_prefix, max_count, select_class_name, select_class_count, test_ratio, seg_image_ratio)