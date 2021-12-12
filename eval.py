import torch 
from net.model import get_model
from data import prepare_data
import fire
import numpy as np
from scipy import stats
from collections import OrderedDict
from util import batch_iou, read_image_tensor, draw_segmentation_mask, get_image_mask, show_cam_on_image
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

BUSI_LABELS = ["malignant", "benign"]
BIRD_LABELS = ['003.Sooty_Albatross', '014.Indigo_Bunting', '067.Anna_Hummingbird', '102.Western_Wood_Pewee', '112.Great_Grey_Shrike', '122.Harris_Sparrow', '188.Pileated_Woodpecker', '194.Cactus_Wren'] 


def mean_confidence_interval(x, confidence=0.95):
    # get CI with 0.95 confidence following normal gaussian distribution
    n = len(x)
    m, se = np.mean(x), stats.sem(x)
    ci = stats.t.ppf((1 + confidence) / 2., n-1) * se
    # ci = 1.96 * se  # assume gaussian distribution
    return m, ci


class Eval():
    def __init__(self,
                 feat_name, 
                 model_weights, 
                 mask_name=None, 
                 num_classes=2, 
                 image_size=224, 
                 device="cpu",
                 dataset="BIRD",
                 multi_gpu=False,
                 num_blocks=3):
        super(Eval, self).__init__()
        self.feat_name = feat_name
        self.mask_name = mask_name
        self.num_classes = num_classes
        self.model_weights = model_weights
        self.image_size=image_size
        self.device=device
        self.dataset = dataset
        self.multi_gpu = multi_gpu
        self.load_model()
        if dataset == "BIRD":
            self.label = BIRD_LABELS
        elif dataset == "BUSI":
            self.label = BUSI_LABELS
        self.num_blocks = num_blocks

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
    
    def image2mask(self, seg_image_list=None, mask_save_file=None, binary_mask=True):
        # load images in the seg_image_list if exists
        # draw mask instead of computing the IOU values or other metrics
        image_df = pd.read_csv(seg_image_list, header=None)
        images = image_df.iloc[:, 0]
        masks = image_df.iloc[:, 2]
        image_list = []
        real_mask_list = []
        for idx in range(len(images)):
            image_tensor = read_image_tensor(images[idx], self.image_size)
            mask = get_image_mask(masks[idx], self.image_size, dataset=self.dataset)
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
            if self.feat_name == "deeplabv3":
                prob = torch.nn.Sigmoid()(outputs)
            pred_mask_tensor = (prob>0.5).type(torch.int)
        else:
            if self.mask_name:
                # interpolate mask to original size
                outputs = torch.nn.functional.interpolate(outputs[1], size=(self.image_size, self.image_size), mode="bicubic")
                pred_mask_tensor = (outputs>0.5).type(torch.int) 
            else:
                _, pred_mask_tensor = torch.max(outputs, 1, keepdim=True)
            # print(torch.max(pred_mask_tensor), torch.max(outputs), outputs)
            pred_mask_tensor = (pred_mask_tensor>0).type(torch.int)
            # pred_mask_tensor = outputs[1].detach()
            # pred_mask_tensor = pred_mask_tensor[pred_mask_tensor > 0.5]
        if binary_mask:
            pred_mask_tensor = (prob>0.5).type(torch.int)
            draw_segmentation_mask(image_tensor, real_mask_tensor, pred_mask_tensor, mask_save_file) 
        else:
            pred_mask_tensor = prob[0] # use first image
            img = (image_tensor[0]+1)/2 # scale to 0-1
            img = img.numpy().transpose([1, 2, 0])
            mask = pred_mask_tensor[0].cpu().detach().numpy()
            # mask = mask / np.max(mask)
            show_cam_on_image(img, mask, mask_save_file, use_rgb=False)
        
    def accuracy(self, test_file=None):
        if test_file is None:
            if self.dataset == "BIRD":
                train_file = "data/bird_train_part.txt"
                test_file = "data/bird_test.txt"
            elif self.dataset == "BUSI":
                train_file = "data/busi_train_part.txt"
                test_file = "data/busi_test.txt"
        else:
            train_file = test_file
        config = {"image_size": self.image_size, 
                  "train": train_file, 
                  "test": test_file, 
                  "dataset": self.dataset}
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        result_matrics = np.zeros((self.num_classes, self.num_classes))
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
        print("Class labels: ", self.label)
        print('Precision: ', res_acc, end=' ')
        print("Mean acc: ", np.mean(res_acc))
        print('Sensitivity: ', res_sens, end=' ')
        print("Mean sensi: ",  np.mean(res_sens))
        print('Specificity: ', res_speci, end=' ')
        print("Mean speci: ", np.mean(res_speci))
        print('F1 score: ', f1_score, end=' ')
        print('Mean F1: ', np.mean(f1_score))          

    def iou(self, test_file=None):
        if test_file is None:
            if self.dataset == "BIRD":
                train_file = "data/bird_train_part.txt"
                test_file = "data/bird_test.txt"
            elif self.dataset == "BUSI":
                train_file = "data/busi_train_part.txt"
                test_file = "data/busi_test.txt"
        else:
            train_file = test_file
        config = {"image_size": self.image_size, 
                "train": train_file, 
                "test": test_file, 
                "dataset": self.dataset}
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

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
                    _, pred_mask_tensor = torch.max(outputs[1], 1, keepdim=True)
                    # print(torch.max(pred_mask_tensor), torch.max(outputs), outputs)
                    pred_mask_tensor = (pred_mask_tensor>0).type(torch.int)
                iou = batch_iou(pred_mask_tensor, mask, 2)
                result_matrics.append(iou[0])
        print("Segmentation IOU: ", np.mean(result_matrics))

    def saliency(self, image_path, target_category=None, saliency_file=None, method="grad-cam"):
        image_tensor = read_image_tensor(image_path, self.image_size)
        try:
            target_layers = [self.model.net[-1][-1]]
        except:
            target_layers = [self.model.net.net[-1][-1]]
        if method == "grad-cam":
            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=False)
        # target_category = [int(target_category)]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=image_tensor, target_category=target_category)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False)
        cv2.imwrite(saliency_file, visualization)
        print("Draw saliency map with {} done! Save in {}".format(method, saliency_file))

if __name__ == "__main__":
    fire.Fire(Eval)