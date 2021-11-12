import torch
import torch.optim as optim 
import torch.nn as nn
from data import prepare_data
from net.model import get_model
import copy 
import time, os
import fire
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from util import batch_iou
from collections import OrderedDict
import re


def train(model, 
          model_name, 
          model_save_path, 
          dataloader, 
          optimizer, 
          criterion, 
          num_epochs, 
          num_classes,
          device="cpu",
          mask_weight=None):
    """Train Classifier"""
    # best_model_w = copy.deepcopy(model.state_dict())
    best_acc = 0
    acc_history = []
    start_t = time.time() 
    max_save_count = 5
    save_counter = 0
    save_interval = 5
    # add log to tensorborad 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    best_test_model = os.path.join(model_save_path, "best_model.pt") 
    if model_name == "resnet50_mask":
        mask_criterion = nn.L1Loss()
        if mask_weight is None:
            mask_weight = 1.0 
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0
            for data in dataloader[phase]:
                inputs = data["image"].to(device)
                labels = data["label"].to(device)
                if model_name in ["deeplabv3", "resnet50_mask"]:
                    masks = data["mask"].to(device)
                    # print(masks, torch.min(masks), torch.max(masks))
                optimizer.zero_grad()
                # if use_cent_loss:
                #     optimizer_cent.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    if model_name == "deeplabv3":
                        # masks = torch.squeeze(masks, dim=1)
                        # masks = masks.permute(1,2,0)
                        # masks = masks.reshape(-1)
                        loss = criterion(outputs, masks)
                        # outputs = outputs.permute(2, 3, 0, 1)
                        # outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1]*outputs.shape[2], outputs.shape[3])
                        # loss = nn.CrossEntropyLoss()(outputs, masks)
                        # _, preds = torch.max(outputs, axis=1)
                        preds = (nn.Sigmoid()(outputs) > 0.5).type(torch.int)
                        # pred_mask = preds.data.cpu().numpy().ravel()
                        # real_mask = masks.data.cpu().numpy().ravel()
                    elif model_name == "resnet50_mask":
                        cls_loss = criterion(outputs[0], labels)
                        # resize masks to final feature size
                        featmap_size = outputs[1].shape[-1]
                        masks_inter = nn.functional.interpolate(masks, size=(featmap_size, featmap_size), mode="bilinear")
                        mask_loss = mask_criterion(outputs[1], masks_inter)
                        loss = cls_loss + mask_loss * mask_weight
                        _, preds = torch.max(outputs[0], 1)
                    else:
                        loss = criterion(outputs[0], labels)
                        _, preds = torch.max(outputs[0], 1)
                        # preds = preds.data.cpu().numpy().ravel()
                        # labels = labels.data.cpu().numpy().ravel()
                    # print(outputs[0], labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                # print(preds, labels.data, torch.sum(preds == labels.data))
                if model_name == "deeplabv3":
                    # match_ratio = np.sum(pred_mask == real_mask) / (input_size ** 2)  # mean via image size
                    ious = batch_iou(preds, masks, 2) # background + foreground
                    running_corrects += np.sum(ious)
                else:
                    running_corrects += np.sum(preds.data.cpu().numpy() == labels.data.cpu().numpy())
            datasize = len(dataloader[phase].dataset)
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize
            print("{} Loss: {:.4f}, Acc{:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_test_model)
            if phase == "test":
                acc_history.append(epoch_acc)
        if not (epoch % save_interval):
            save_counter += 1
            torch.save(model.state_dict(), model_save_path+"/w_epoch_{}.pt".format(epoch+1))
            oldest = epoch+1-save_interval*max_save_count
            if os.path.exists(model_save_path+"/w_epoch_{}.pt".format(oldest)):
                os.remove(model_save_path+"/w_epoch_{}.pt".format(oldest))
    time_elapsed = time.time() - start_t 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best val acc: {:.4f}".format(best_acc))
    # model.load_state_dict(torch.load(best_model_w, map_location=torch.device(device)))
    return model, acc_history

def load_weights(model, pretrained_weights, multi_gpu=False, device="cpu", num_classes=3):
    state_dict=torch.load(pretrained_weights, map_location=torch.device(device))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): # different class number, drop the last fc layer
        if re.search("fc", k):
            w_shape = list(v.size())
            # drop normal class weights if the number of classes of pretrained weights is different from current training setting.
            if w_shape != num_classes:
                w_shape = [num_classes, *w_shape[1:]]
                v = torch.rand(w_shape, requires_grad=True)
                if len(w_shape) > 2:
                    nn.init.kaiming_normal_(v) # weights
                else:
                    v.data.fill_(0.01) # bias
        if multi_gpu:
            name = k[7:] # remove 'module.' of dataparallel
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def run(model_name, 
        image_size=448, 
        num_classes=3, 
        batch_size=32, 
        num_epochs=40, 
        model_save_path="train_res", 
        device="cuda:0", 
        lr=0.001, 
        moment=0.9, 
        use_pretrained=True,
        pretrained_weights=None,
        dataset="BUSI",
        num_gpus=1, 
        dilute_mask=0,
        mask_weight=None):
    # get model 
    model = get_model(model_name=model_name,
                      num_classes=num_classes, 
                      use_pretrained=use_pretrained, 
                      return_logit=False).to(device)
    # load pretrained model weights
    if pretrained_weights: 
        try:
            model = load_weights(model, pretrained_weights, multi_gpu=False, device=device, num_classes=num_classes)
        except:
            model = load_weights(model, pretrained_weights, multi_gpu=True, device=device, num_classes=num_classes)
    if dataset == "BUSI":
        train_file = "data/train_sample.txt"
        test_file = "data/test_sample.txt"
    elif dataset == "test_BUSI": 
        train_file = "example/debug_BUSI.txt"
        test_file = "example/debug_BUSI.txt"
        dataset = "BUSI"
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        # deploy model on multi-gpus
        model = nn.DataParallel(model, device_ids=device_ids)
    config = {"image_size": image_size, 
              "train": train_file, 
              "test": test_file, 
              "dataset": dataset,
              "mask": model_name in ["deeplabv3", "resnet50_mask"],
              "dilute_mask": dilute_mask,
              }
    image_datasets, data_sizes = prepare_data(config)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=x=="train", batch_size=batch_size, num_workers=0,   drop_last=True) for x in ["train", "test"]}
    # loss function
    if dataset == "BUSI":
        cls_weight = [2.0, 1.0, 1.0]
    if model_name == "deeplabv3":
        # criterion = nn.NLLLoss(reduction="mean").to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0)).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weight)).to(device)
    # optimizer
    print("optimized parameter names")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    print("-"*40)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=moment) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # if use_cent_loss:
    #     criterion_cent = CenterLoss(num_classes, feat_dim=feat_dim).to(device)
    #     optim_cent = torch.optim.SGD(criterion_cent.parameters(), lr=lr_cent)
    # else:
    #     criterion_cent, optim_cent = None, None
    model_ft, hist = train(model=model, 
                           model_name=model_name, 
                           model_save_path=model_save_path, 
                           dataloader=dataloaders, 
                           optimizer=optimizer, 
                           criterion=criterion, 
                           num_epochs=num_epochs, 
                           num_classes=num_classes,
                           device=device,
                           mask_weight=mask_weight)
    # torch.save(model_ft.state_dict(), model_save_path+'/best_model.pt')
    print("Val acc history: ", hist)

if __name__ == "__main__":
    fire.Fire(run)
    # # training config
    # input_size = 224
    # num_classes = 3 
    # batch_size = 16
    # num_epoches = 40
    # model_name = "resnet50"
    # device = "cuda:0"
    # input_dir = "/shared/anastasio5/COVID19/data/covidx"
    # model_save_path = "covidx_res50"

