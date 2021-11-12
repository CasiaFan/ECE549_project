import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import glob, os 
import pandas as pd 
# from skimage import io 
import numpy as np 
import re
import random
from util import get_image_mask, dilute_mask, parse_mayo_mask_box


BUSI_LABELS = ["normal", "malignant", "benign"] # BUSI dataset labels: https://bcdr.eu/information/downloads

# TODO: add gaussian noise or white noise to mask 
class BUSI_dataset(Dataset):
    def __init__(self, csv_file, transform=None, mask=False, mask_transform=None, mask_dilute=15):
        """
        csv_file: csv file containing image file path and corresponding label
        transform: transform for image
        mask: return mask or not
        mask_transform: transformation for mask
        mask_dilute: dilute mask with given distance in all directions
        """
        df = pd.read_csv(csv_file, sep=",", header=None)
        df.columns = ["img", "label"]
        self._img_files = df["img"].tolist()
        self._img_labels = df["label"].tolist()
        self._transform = transform
        self._mask = mask
        self._mask_transform = mask_transform
        self._mask_dilute = mask_dilute

    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        image_name = self._img_files[idx]
        assert os.path.exists(image_name), "Image file not found!"
        # load image
        img = Image.open(image_name)
        img = img.convert("RGB")
        label = self._img_labels[idx]
        label_id = BUSI_LABELS.index(label)
        #onehot_id = torch.nn.functional.one_hot(torch.Tensor(label_id), len(LABELS))
        # get the identical random seed for both image and mask
        seed = random.randint(0, 2147483647)
        if self._transform:
            # state = torch.get_rng_state()
            random.seed(seed)
            torch.manual_seed(seed)
            img = self._transform(img)
        # load mask
        mask = []
        if self._mask:
            mask = get_image_mask(image_name, dataset="BUSI")
            mask = dilute_mask(mask, dilute_distance=self._mask_dilute)
            # assign class label
            mask = Image.fromarray(mask)
            if self._mask_transform:
                random.seed(seed)
                torch.manual_seed(seed)
                # torch.set_rng_state(state)
                mask = self._mask_transform(mask)
                mask = mask.type(torch.float)
                # mask = mask * label_id  # normal case is identical to backaground
        return {"image": img, "label": label_id, "mask": mask}


# input image width/height ratio
BUSI_IMAGE_RATIO = 570/460


def prepare_data(config):
    """
    config: 
        image_size: size of input images
        train: train img file list
        test: test img file list
        dataset: name of dataset to use: BUSI
    """
    data_transforms = {
        'train_image': transforms.Compose([   
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            # transforms.Resize(config["image_size"]),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(config["image_size"], scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.12, 0.12, 0.08, 0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'train_mask': transforms.Compose([   
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            # transforms.Resize(config["image_size"]),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(config["image_size"], scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.12, 0.12, 0.08, 0.08),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), 
        'test_image': transforms.Compose([
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test_mask': transforms.Compose([
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), 
    }
    mask = config.get("mask", None)
    if config["dataset"] == "BUSI":
        image_datasets = {x: BUSI_dataset(config[x], 
                                          transform=data_transforms[x+"_image"], 
                                          mask=mask, 
                                          mask_transform=data_transforms[x+"_mask"], 
                                          mask_dilute=config.get("mask_dilute", 0)) for x in ["train", "test"]}
    else:
        print("Unknown dataset")
    # class_names = image_datasets["train"].classes 
    data_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    return image_datasets, data_sizes


def generate_image_list(img_dir, save_dir, test_sample_size=40):
    global BUSI_LABELS
    image_list = glob.glob(img_dir+"/**/*.bmp", recursive=True)
    train_sample_file = os.path.join(save_dir, "train_sample_test.txt")
    test_sample_file = os.path.join(save_dir, "test_sample_test.txt")
    random.shuffle(image_list)
    train_f = open(train_sample_file, "w")
    test_f = open(test_sample_file, "w")
    counter = {x: 0 for x in BUSI_LABELS}
    for img in image_list:
        class_name = os.path.basename(os.path.dirname(img))
        # ignore mask images
        if re.search("mask", img):
            continue
        if counter[class_name] > test_sample_size:
            write_f = train_f 
        else:
            write_f = test_f 
        write_f.write("{},{}\n".format(img, class_name))
        counter[class_name] += 1
    train_f.close()
    test_f.close()


if __name__ == "__main__":
    import cv2
    from util import draw_segmentation_mask
    image_dir = "/Users/zongfan/Projects/data/originals"
    # image_dir = "/Users/zongfan/Projects/data/Dataset_BUSI_with_GT"
    config = {"image_size": 224, "train": "train_sample.txt", "test": "test_sample.txt", "dataset": "BUSI", "mask": True}
    ds, _ = prepare_data(config)
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(ds["train"], batch_size=batch_size)
    for data in dataloader:
        imgs = data['image']
        masks = data["mask"]
        print(imgs.shape)
        img_shape = imgs.shape
        imgs = imgs.view(img_shape[0]*img_shape[1], img_shape[2], img_shape[3], img_shape[4])
        print(imgs.shape)
        draw_segmentation_mask(imgs, masks, "test/test.png")
        break
        for i in range(len(imgs)):
            img = imgs[i].numpy()
            img = (img + 1.) / 2. * 255
            mask = masks[i].numpy()
            mask = mask.repeat(3, axis=0)
            mask = mask * 255
            img = np.concatenate([img, mask], axis=2)
            x = np.transpose(img, (1, 2, 0))
            x = x.astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            cv2.imshow("test", x)
            if cv2.waitKey(0) == ord("q"):
                exit()
    generate_image_list(image_dir, ".", test_sample_size=32)

