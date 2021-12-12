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


BUSI_LABELS = ["malignant", "benign"] # BUSI dataset labels: https://bcdr.eu/information/downloads
BIRD_LABELS =  ['003.Sooty_Albatross', '014.Indigo_Bunting', '067.Anna_Hummingbird', '102.Western_Wood_Pewee', '112.Great_Grey_Shrike', '122.Harris_Sparrow', '188.Pileated_Woodpecker', '194.Cactus_Wren'] 


class MaskDataset(Dataset):
    def __init__(self, csv_file, dataset_name="BIRD", transform=None, mask_transform=None, mask_dilute=0, image_size=224):
        """
        csv_file: csv file containing image file path and corresponding label
        transform: transform for image
        mask_transform: transformation for mask
        mask_dilute: dilute mask with given distance in all directions
        """
        df = pd.read_csv(csv_file, sep=",", header=None)
        df.columns = ["img", "label", "mask"]
        self._img_files = df["img"].tolist()
        self._img_labels = df["label"].tolist()
        self._img_masks = df["mask"].tolist()
        self._transform = transform
        self._mask_transform = mask_transform
        self._mask_dilute = mask_dilute
        self._img_size = image_size
        self._dataset_name = dataset_name
        if dataset_name == "BIRD":
            self._label = BIRD_LABELS
        elif dataset_name == "BUSI":
            self._label = BUSI_LABELS
        else:
            print("Unknown dataset name!")

    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        image_name = self._img_files[idx]
        assert os.path.exists(image_name), "Image file not found!"
        # load image
        img = Image.open(image_name)
        img = img.convert("RGB")
        label = self._img_labels[idx]
        label_id = self._labels.index(label)
        #onehot_id = torch.nn.functional.one_hot(torch.Tensor(label_id), len(LABELS))
        # get the identical random seed for both image and mask
        seed = random.randint(0, 2147483647)
        if self._transform:
            # state = torch.get_rng_state()
            random.seed(seed)
            torch.manual_seed(seed)
            img = self._transform(img)
        # load mask
        if self._img_masks[idx] != "none":
            mask = get_image_mask(self._img_masks[idx], dataset=self._dataset_name)
            mask = dilute_mask(mask, dilute_distance=self._mask_dilute)
            # assign class label
            mask = Image.fromarray(mask)
            if_mask = True
            if self._mask_transform:
                random.seed(seed)
                torch.manual_seed(seed)
                # torch.set_rng_state(state)
                mask = self._mask_transform(mask)
                mask = mask.type(torch.float)
                # mask = mask * label_id  # normal case is identical to backaground
        else:
            mask = torch.zeros((1, self._img_size, self._img_size))
            if_mask = False
        return {"image": img, "label": label_id, "mask": mask, "if_mask": if_mask}
    

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
    image_datasets = {x: MaskDataset(config[x], dataset_name=config["dataset"], transform=data_transforms[x+"_image"],
                                          mask_transform=data_transforms[x+"_mask"]) for x in ["train", "test"]}

    # class_names = image_datasets["train"].classes 
    data_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    return image_datasets, data_sizes


if __name__ == "__main__":
    import cv2
    from util import draw_segmentation_mask
    config = {"image_size": 224, "train": "data/bird_train_part.txt", "test": "data/bird_test_part.txt", "dataset": "BIRD", "mask": True}
    ds, _ = prepare_data(config)
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(ds["test"], batch_size=batch_size)
    for data in dataloader:
        imgs = data['image']
        masks = data["mask"]
        print(data["if_mask"])
        print(imgs.shape)
        draw_segmentation_mask(imgs, masks, masks, "test/test.png")
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

