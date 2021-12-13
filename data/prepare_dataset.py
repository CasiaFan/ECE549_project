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
        random.seed(42)
        random.shuffle(class_dirs)
        select_class_name = class_dirs[:select_class_count]
        print("selected_classes: ", sorted(select_class_name))
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
    train_labels = [os.path.dirname(x).split("/")[-1] for x in train_images]
    test_labels = [os.path.dirname(x).split("/")[-1] for x in test_images]
    train_seg_idx = random.sample(list(range(train_img_count)), int(train_img_count * seg_image_ratio))
    test_seg_idx = random.sample(list(range(test_img_count)), int(test_img_count * seg_image_ratio))
    train_segs = ["none" for _ in range(train_img_count)]
    test_segs = ["none" for _ in range(test_img_count)]
    # for idx in train_seg_idx:
    for idx in range(train_img_count):
        image_name = os.path.basename(train_images[idx])
        image_class = os.path.dirname(train_images[idx]).split("/")[-1]
        seg_name = os.path.join(seg_dir, image_class, image_name.replace("jpg", "png"))
        train_segs[idx] = seg_name
    # for idx in test_seg_idx:
    for idx in range(test_img_count):
        image_name = os.path.basename(test_images[idx])
        image_class = os.path.dirname(test_images[idx]).split("/")[-1]
        seg_name = os.path.join(seg_dir, image_class, image_name.replace("jpg", "png"))
        test_segs[idx] = seg_name 
    print("=======write=========")
    print("Num of train images: {}".format(train_img_count))
    print("Num of test images: {}".format(test_img_count))
    with open(train_file, "w") as f:
        for img, lbl, seg in zip(train_images, train_labels, train_segs):
            f.write(img + "," + lbl + "," + seg + "\n")
    f.close()
    with open(test_file, "w") as f:
        for img, lbl, seg in zip(test_images, test_labels, test_segs):
            f.write(img + "," + lbl + "," + seg + "\n")
    f.close() 


def filter_busi_dataset(image_dir, save_dir, 
                        save_name_prefix, 
                        max_count=250,
                        select_class_name=None, 
                        test_ratio=0.2,
                        seg_image_ratio=0.25,
                        ):
    os.makedirs(save_dir, exist_ok=True)
    class_dirs = os.listdir(image_dir)
    train_file = os.path.join(save_dir, save_name_prefix+"_train.txt")
    test_file = os.path.join(save_dir, save_name_prefix+"_test.txt")
    # random select n classes
    if (select_class_name is None) and (select_class_count is not None) and isinstance(select_class_count, int):
        random.seed(42)
        random.shuffle(class_dirs)
        select_class_name = class_dirs[:select_class_count]
        print("selected_classes: ", sorted(select_class_name))
    images = []
    for c in select_class_name:
        c_images = glob.glob(os.path.join(image_dir, c, "*_mask.png"), recursive=True)
        if len(c_images) > max_count:
            random.shuffle(c_images)
            c_images = c_images[:max_count]
        images += c_images
    images = [image.replace("_mask", "") for image in images]
    train_images, test_images = train_test_split(images, test_size=test_ratio, shuffle=True)
    # select 25% images to extract segmentation data
    train_img_count = len(train_images)
    test_img_count = len(test_images)
    train_labels = [os.path.dirname(x).split("/")[-1] for x in train_images]
    test_labels = [os.path.dirname(x).split("/")[-1] for x in test_images]
    train_seg_idx = random.sample(list(range(train_img_count)), int(train_img_count * seg_image_ratio))
    # test_seg_idx = random.sample(list(range(test_img_count)), int(test_img_count * seg_image_ratio))
    train_segs = ["none" for _ in range(train_img_count)]
    test_segs = ["none" for _ in range(test_img_count)]
    for idx in train_seg_idx:
    #for idx in range(train_img_count):
        seg_name = train_images[idx].replace(".png", "_mask.png")
        train_segs[idx] = seg_name
    # for idx in test_seg_idx:
    for idx in range(test_img_count):
        seg_name = test_images[idx].replace(".png", "_mask.png")
        test_segs[idx] = seg_name 
    print("=======write=========")
    print("Num of train images: {}".format(train_img_count))
    print("Num of test images: {}".format(test_img_count))
    with open(train_file, "w") as f:
        for img, lbl, seg in zip(train_images, train_labels, train_segs):
            f.write(img + "," + lbl + "," + seg + "\n")
    f.close()
    with open(test_file, "w") as f:
        for img, lbl, seg in zip(test_images, test_labels, test_segs):
            f.write(img + "," + lbl + "," + seg + "\n")
    f.close() 


if __name__ == "__main__":
    #image_dir="/home/zongfan2/Documents/ECE549_project/CUB_200_2011/CUB_200_2011/images" 
    #seg_dir="/home/zongfan2/Documents/ECE549_project/bird_seg"
    #save_dir="/home/zongfan2/Documents/ECE549_project/ECE549_project/data"
    #save_name_prefix="bird"
    #max_count=100
    #select_class_name=None
    #select_class_count=8
    #test_ratio=0.2
    #seg_image_ratio=0.25
    #filter_bird_dataset(image_dir, seg_dir, save_dir, save_name_prefix, max_count, select_class_name, select_class_count, test_ratio, seg_image_ratio)

    image_dir="/shared/anastasio5/COVID19/data/Dataset_BUSI_with_GT" 
    save_dir="/home/zongfan2/Documents/ECE549_project/ECE549_project/data"
    save_name_prefix="busi"
    max_count=250
    select_class_name=["malignant", "benign"]
    test_ratio=0.2
    seg_image_ratio=0.25
    filter_busi_dataset(image_dir, save_dir, save_name_prefix, max_count, select_class_name, test_ratio, seg_image_ratio)
