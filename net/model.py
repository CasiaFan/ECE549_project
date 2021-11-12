import torch
import torch.nn as nn
from torchvision import models
try:    
    from .seg_net import DeepLabV3
except:
    from seg_net import DeepLabV3


def initialize_weights(net):
    for l in net.modules():
        if isinstance(l, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(l.weight)
            # l.bias.data.fill_(0)
        if isinstance(l, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            nn.init.constant_(l.weight, 1.0)
            nn.init.constant_(l.bias, 0.0)

class LogitResnet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, return_logit=False, use_pretrained=True, return_feature=False):
        super(LogitResnet, self).__init__()
        if model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=use_pretrained)
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=use_pretrained)
        else:
            print("unknown resnet model")
            exit()
        num_features = model.fc.in_features
        self.return_logit = return_logit
        self.return_feature = return_feature
        self.net = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, inputs):
        f = self.net(inputs)
        x = self.avgpool(f)
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit and self.return_feature:
            return [x, f, l]
        if self.return_logit:
            return [x, l]
        if self.return_feature:
            return [x, f]
        return [x]

class LogitDensenet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, return_feature=False, return_logit=False, use_pretrained=True):
        super(LogitDensenet, self).__init__()
        if model_name == "densenet161":
            model = models.densenet161(pretrained=use_pretrained)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=use_pretrained)
        else:
            print("unknown densenet structure")
        num_features = model.classifier.in_features
        self.return_logit = return_logit
        self.return_feature = return_feature
        self.fc = nn.Linear(num_features, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, inputs):
        x = self.net(inputs)
        f = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit and self.return_feature:
            return [x, f, l]
        if self.return_feature:
            return [x, f]
        if self.return_logit:
            return [x, l]
        return [x]


class AttentionBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MaskAttentionNet(nn.Module):
    """
    Learn mask attention map supervised by real masks 
    Augment features by lambda*(sigma(f)*f)+f, where f is input feature, 
    sigma(x) simulates the real mask information, lambda is attention weight
    """
    def __init__(self, reduction='mean', attention_weight=0.25):
        """
        reduction: method to reduce C channels into 1, 'mean' or 'max'
        """
        super(MaskAttentionNet, self).__init__()
        # self.bn = nn.BatchNorm2d(1)
        # self.relu = nn.LeakyReLU(0.02)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.act = nn.Sigmoid()
        self.reduction = reduction
        # self.attention_weight = attention_weight
        initialize_weights(self)

    def forward(self, x):
        # pool input feature from (N, C, H, W) to (N, 1, H, W)
        if self.reduction == "mean":
            x_pool = torch.mean(x, 1, keepdim=True)
        elif self.reduction == "max":
            x_pool, _ = torch.max(x, 1, keepdim=True)
        # x_map = self.bn(x_pool)
        # x_map = self.relu(x_map)
        x_map = self.conv(x_pool)
        x_map = self.act(x_map)
        # aug_f = x + self.attention_weight * x_map * x
        return x_map

class MaskAttentionNet2(nn.Module):
    """
    Optimized attention mask with continuous attention blocks which shink the channel from 2048 to 1
    """
    def __init__(self, num_blocks=4):
        """
        reduction: method to reduce C channels into 1, 'mean' or 'max'
        """
        super(MaskAttentionNet2, self).__init__()
        init_channels = 2048
        channel_list = [init_channels//2**i for i in range(num_blocks)]
        channel_list.append(1)
        att_blocks = []
        for i in range(len(channel_list)-1):
            att_blocks.append(AttentionBlock(channel_list[i], channel_list[i+1]))
        self.net = nn.Sequential(*att_blocks)
        self.act = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x_map = self.net(x)
        x_map = self.act(x_map)
        return x_map

class ClassificationHead(nn.Module):
    def __init__(self, inchannels, num_classes):
        super(ClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inchannels, num_classes)
        initialize_weights(self)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class RasaeeUpsampleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, output_size):
        super(RasaeeUpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6(inplace=True)
        self.up = nn.Upsample([output_size, output_size], mode="bilinear", align_corners=True)
        self.bn = nn.BatchNorm2d(output_channel)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.bn(x)
        return x

class RasaeeMaskHead(nn.Module):
    """https://arxiv.org/abs/2108.04345"""
    def __init__(self, image_size=448):
        super(RasaeeMaskHead, self).__init__()
        self.image_size = image_size
        self.block1 = RasaeeUpsampleBlock(2048, 256, 16)
        self.block2 = RasaeeUpsampleBlock(256, 64, 112)
        self.block3 = RasaeeUpsampleBlock(64, 1, image_size)
        self.sig = nn.Sigmoid()
        initialize_weights(self)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.sig(x)
        return x 

class ResNetMask(nn.Module):
    """Resnet with mask attention module"""
    def __init__(self,  
                 model_name,
                 num_classes, 
                 use_pretrained=True, 
                 num_blocks=3,
                 reduction='mean', 
                 attention_weight=0.5,
                 image_size=448):
        super(ResNetMask, self).__init__()
        model_info = model_name.split("_")
        resnet_name = model_info[0]
        self.attention = model_info[1] == "attention"
        self.attention_weight = attention_weight
        self.net = LogitResnet(resnet_name, num_classes, return_logit=False, return_feature=True, use_pretrained=use_pretrained)
        if model_name == "resnet50_attention_mask":
            # self.mask_module = MaskAttentionNet(reduction=reduction, attention_weight=attention_weight)
            self.mask_module = MaskAttentionNet2(num_blocks)
        elif model_name == "resnet50_rasaee_mask":
            self.mask_module = RasaeeMaskHead(image_size)
        self.c = ClassificationHead(2048, num_classes)

    def forward(self, x):
        _, x = self.net(x)
        mask = self.mask_module(x)
        if self.attention:
            # x = x + self.attention_weight * mask * x
            x = x * mask
        x = self.c(x)
        return x, mask


def get_model(model_name, 
              num_classes, 
              use_pretrained=True, 
              return_logit=False, 
              return_feature=False, 
              reduction='mean', 
              attention_weight=0.25,
              image_size=448,
              num_blocks=3,
              **kwargs):
    if model_name in ["resnet50", "resnet34", "resnet18"]:
        model = LogitResnet(model_name, num_classes, return_logit=return_logit, use_pretrained=use_pretrained, return_feature=return_feature)
    elif model_name == "vgg":
        model = models.vgg16_bn(pretrained=use_pretrained)
        in_features = 25088
        model.classifer = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif model_name in ["densenet121", "densenet161"]:
        model = LogitDensenet(model_name, num_classes, return_logit=return_logit, return_feature=return_feature, use_pretrained=use_pretrained)
    elif model_name == "inception_v3":
        print("Warning: Inception V3 input size must be larger than 300x300")
        if use_pretrained:
            model = models.inception_v3(pretrained=True, aux_logits=False)
        else:
            model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "deeplabv3":
        model = DeepLabV3(num_classes=num_classes, pretrained=use_pretrained)
    elif model_name == "resnet50_attention_mask":
        model = ResNetMask(model_name, num_classes, use_pretrained=use_pretrained, reduction=reduction, attention_weight=attention_weight, num_blocks=num_blocks)
    elif model_name == "resnet50_rasaee_mask":
        model = ResNetMask(model_name, num_classes, use_pretrained=use_pretrained, image_size=image_size)
    else:
        print("unknown model name!")
    return model

# def decoder()

if __name__ == "__main__":
    inputs = torch.rand(2, 3, 224, 224)
    # inputs = torch.rand(2, 3, 32, 32) 
    # model = get_model("resnet50", 3, use_pretrained=False, return_logit=True)
    # model = models.densenet161(pretrained=False, num_classes=3) 
    model = get_model("resnet50_attention_mask", 3, use_pretrained=False, return_logit=True, return_feature=True) 
    # print(list(model.children())[:-1])
    # model = nn.Sequential(*list(model.children())[:-1])
    res = model(inputs)
    print(res[0].shape, res[1].shape)
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t", name, param.shape)
    # print("-"*10)   

39,2, 4, 35

