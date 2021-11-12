import torch 
from net.model import get_model
from data import prepare_data
import fire
import numpy as np
from scipy import stats
from collections import OrderedDict
from util import batch_iou, read_image_tensor, draw_segmentation_mask, get_image_mask, show_cam_on_image
import pandas as pd


BUSI_LABELS = ["normal", "malignant", "benign"]


def mean_confidence_interval(x, confidence=0.95):
    # get CI with 0.95 confidence following normal gaussian distribution
    n = len(x)
    m, se = np.mean(x), stats.sem(x)
    ci = stats.t.ppf((1 + confidence) / 2., n-1) * se
    # ci = 1.96 * se  # assume gaussian distribution
    return m, ci


class Eval():
    def __init__(self, model_name, 
                 num_classes, 
                 model_weights,  
                 image_size=224, 
                 device="cpu",
                 dataset="covidx",
                 multi_gpu=False,):
        super(Eval, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model_weights = model_weights
        self.image_size=image_size
        self.device=device
        self.dataset = dataset
        self.multi_gpu = multi_gpu
        self.load_model()
    
    def load_model(self):
        self.model = get_model(model_name=self.model_name, 
                          num_classes=self.num_classes, 
                          use_pretrained=True, 
                          return_logit=False).to(self.device)
        state_dict=torch.load(self.model_weights, map_location=torch.device(self.device))
        if self.multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def image2mask(self, seg_image_list=None, mask_save_file=None):
        # load images in the seg_image_list if exists
        # draw mask instead of computing the IOU values or other metrics
        image_df = pd.read_csv(seg_image_list, header=None)
        images = image_df.iloc[:, 0]
        image_list = []
        real_mask_list = []
        for image in images:
            image_tensor = read_image_tensor(image, self.image_size)
            mask = get_image_mask(image, self.image_size, dataset="BUSI")
            # mask = mask / 255
            mask = np.expand_dims(mask, 0)
            mask = torch.tensor(mask)
            real_mask_list.append(mask)
            image_list.append(image_tensor)
        image_tensor = torch.stack(image_list).to(self.device)
        real_mask_tensor = torch.stack(real_mask_list)
        image_tensor = image_tensor.squeeze(1)
        outputs = self.model(image_tensor)
        if self.num_classes == 1:
            if self.model_name == "deeplabv3":
                prob = torch.nn.Sigmoid()(outputs)
            pred_mask_tensor = (prob>0.5).type(torch.int)
        else:
            if self.model_name in ["resnet50_attention_mask", "resnet50_rasaee_mask"]:
                # interpolate mask to original size
                outputs = torch.nn.functional.interpolate(outputs[1], size=(self.image_size, self.image_size), mode="bicubic")
                pred_mask_tensor = (outputs>0.5).type(torch.int) 
            else:
                _, pred_mask_tensor = torch.max(outputs, 1, keepdim=True)
            # print(torch.max(pred_mask_tensor), torch.max(outputs), outputs)
            pred_mask_tensor = (pred_mask_tensor>0).type(torch.int)
            # pred_mask_tensor = outputs[1].detach()
            # pred_mask_tensor = pred_mask_tensor[pred_mask_tensor > 0.5]
        draw_segmentation_mask(image_tensor, real_mask_tensor, pred_mask_tensor, mask_save_file) 
        # img = image_tensor[0].numpy().transpose([1, 2, 0])
        # mask = pred_mask_tensor[0].numpy()
        # mask = mask / np.max(mask)
        # show_cam_on_image(img, mask, mask_save_file, use_rgb=True)
        
    def accuracy(self, test_file=None):
        if test_file is None:
            if self.dataset == "BUSI":
                train_file = "data/train_sample.txt"
                test_file = "data/test_sample.txt"
            elif self.dataset == "test":
                train_file = "example/debug_sample_benign.txt"
                test_file = "example/debug_sample_benign.txt"
        else:
            train_file = test_file
        config = {"image_size": self.image_size, 
                  "train": train_file, 
                  "test": test_file, 
                  "dataset": self.dataset}
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        # result matrics:
        #      __________________________________________
        #      | gt \ pred | Normal | COVID | Pneumonia | 
        #      ------------------------------------------
        #      |  Normal   |        |       |           |
        #      |  COVID    |        |       |           |
        #      |  Pneumonia|        |       |           |
        #      ------------------------------------------
        if self.dataset == "BUSI":
            result_matrics = np.zeros((3, 3))
        with torch.no_grad():
            for data in dataloader:
                inputs = data["image"].to(self.device)
                labels = data["label"].to(self.device)
                tag = labels.cpu().numpy()[0]
                outputs = self.model(inputs)
                _, pred = torch.max(outputs[0], 1)
                # score = outputs[0].numpy()
                pred = int(pred.item())
                result_matrics[tag][pred] += 1

        # if self.dataset == "BUSI":
        #     result_matrics = np.zeros((3, 3))
        #     with torch.no_grad():
        #         for data in dataloader:
        #             tag = data["label"].data.cpu().numpy()[0]
        #             img = data["image"].to(self.device)
        #             outputs = self.model(img)
        #             _, pred = torch.max(outputs[0], 1)
        #             pred = int(pred.cpu().numpy()[0])
        #             result_matrics[tag][pred] += 1
            # precision: TP / (TP + FP)
            print("result matrics: ", result_matrics)
            # res_acc = [result_matrics[i, i]/np.sum(result_matrics[:,i]) for i in range(num_classes)]
            res_acc = []
            # sensitivity: TP / (TP + FN)
            res_sens = []
            # res_sens = [result_matrics[i, i]/np.sum(result_matrics[i,:]) for i in range(num_classes)]
            # specificity: TN / (TN+FP)
            res_speci = []
            # f1 score: 2TP/(2TP+FP+FN)
            f1_score = []
            for i in range(self.num_classes):
                TP = result_matrics[i,i]
                FN = np.sum(result_matrics[i,:])-TP
                spe_matrics = np.delete(result_matrics, i, 0)
                FP = np.sum(spe_matrics[:, i])
                TN = np.sum(spe_matrics) - FP
                acc = TP/(TP+FP)
                sens = TP/(TP+FN)
                speci = TN/(TN+FP)
                f1 = 2*TP/(2*TP+FP+FN)
                res_acc.append(acc)
                res_speci.append(speci)
                res_sens.append(sens)
                f1_score.append(f1)
        if self.dataset == "BUSI":
            print('Precision: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(res_acc[0],res_acc[1],res_acc[2], np.mean(res_acc)))
            print('Sensitivity: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(res_sens[0],res_sens[1],res_sens[2], np.mean(res_sens)))
            print('Specificity: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(res_speci[0],res_speci[1],res_speci[2], np.mean(res_speci)))
            print('F1 score: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(f1_score[0],f1_score[1],f1_score[2], np.mean(f1_score)))          
        else:
            print("unknown dataset") 
    
    def iou(self, test_file=None):
        if test_file is None:
            if self.dataset == "BUSI":
                train_file = "data/train_sample.txt"
                test_file = "data/test_sample.txt"
            elif self.dataset == "test":
                train_file = "example/debug_sample_benign.txt"
                test_file = "example/debug_sample_benign.txt"
                self.dataset = "BUSI"
        else:
            train_file = test_file
        config = {"image_size": self.image_size, 
                "train": train_file, 
                "test": test_file, 
                "dataset": self.dataset, 
                "mask": True}
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        if self.dataset == "BUSI":
            result_matrics = []
            with torch.no_grad():
                for data in dataloader:
                    img = data["image"].to(self.device)
                    outputs = self.model(img)
                    mask = data["mask"].to(self.device)
                    if self.num_classes == 1:
                        prob = torch.nn.Sigmoid()(outputs)
                        pred_mask_tensor = (prob>0.5).type(torch.int)
                    else:
                        _, pred_mask_tensor = torch.max(outputs, 1, keepdim=True)
                        # print(torch.max(pred_mask_tensor), torch.max(outputs), outputs)
                        pred_mask_tensor = (pred_mask_tensor>0).type(torch.int)
                    iou = batch_iou(pred_mask_tensor, mask, 2)
                    result_matrics.append(iou[0])
            print("Segmentation IOU: ", np.mean(result_matrics))

if __name__ == "__main__":
    fire.Fire(Eval)