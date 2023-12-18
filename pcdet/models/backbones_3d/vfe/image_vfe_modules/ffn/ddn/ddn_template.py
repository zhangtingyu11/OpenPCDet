from collections import OrderedDict
from pathlib import Path
from torch import hub

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        self.model.backbone.return_layers = {
            feat_extract_layer: 'features',
            **self.model.backbone.return_layers
        }

    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        # Get model
        model = constructor(pretrained=False,
                            pretrained_backbone=False,
                            num_classes=self.num_classes,
                            aux_loss=self.aux_loss)

        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            
            # Download pretrained model if not available yet
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict,
                                                          pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        model_num_classes = model_dict["classifier.4.weight"].shape[0]
        pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop("classifier.4.weight")
            pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        """
        对于CaDDN来说, 输入是尺寸为[batch_size, 3, H, W]的图像
        下采样层:
            先使用7*7的步长为2, padding为3的2D卷积(3->64)来处理图像, 输出为[batch_size, 64, H/2, W/2]的特征图
            再对输出的特征图使用BatchNorm2D, ReLU
            对输出的特征图使用3*3的步长为2, padding为1的maxpooling, 输出为[batch_size, 64, H/4, W/4]的特征图
        layer1: 包含三个Bottoleneck层, 输入为[batch_size, 64, H/4, W/4]的特征图, 记为x
            BottleNeck0:
                1*1步长为1的卷积(64->64), BN2D, ReLU
                3*3步长为1, padding为1的卷积(64->64), BN2D, ReLU
                1*1步长为1的卷积(64->256), BN2D, 输出记为out
                然后将x送入1*1的步长为1的卷积(64->256), BN2D中, 输出记为identity
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 256, H/4, W/4]
            BottleNeck1: 输入是上一层的输出, 记为identity
                1*1步长为1的卷积(256->64), BN2D, ReLU
                3*3步长为1, padding为1的卷积(64->64), BN2D, ReLU
                1*1步长为1的卷积(64->256), BN2D, 输出记为out
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 256, H/4, W/4]
            BottleNeck2:
                同BottleNeck1
        layer2: 包含四个Bottoleneck层, 输入为[batch_size, 256, H/4, W/4]的特征图, 记为x
            BottleNeck0:
                1*1步长为1的卷积(256->128), BN2D, ReLU
                3*3步长为2, padding为1的卷积(128->128), BN2D, ReLU
                1*1步长为1的卷积(128->512), BN2D, 输出记为out
                然后将x送入1*1的步长为1的卷积(256->512), BN2D中, 输出记为identity
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 512, H/8, W/8]
            BottleNeck1: 输入是上一层的输出, 记为identity
                1*1步长为1的卷积(512->128), BN2D, ReLU
                3*3步长为1, padding为1的卷积(128->128), BN2D, ReLU
                1*1步长为1的卷积(128->512), BN2D, 输出记为out
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 512, H/8, W/8]
            BottleNeck2 ~ BottleNeck3:
        layer3: 包含23个Bottoleneck层, 输入为[batch_size, 512, H/8, W/8]的特征图, 记为x
            BottleNeck0:
                1*1步长为1的卷积(512->256), BN2D, ReLU
                3*3步长为1, padding为1的卷积(256->256), BN2D, ReLU
                1*1步长为1的卷积(256->1024), BN2D, 输出记为out
                然后将x送入1*1的步长为1的卷积(512->1024), BN2D中, 输出记为identity
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 1024, H/8, W/8]
            BottleNeck1:
                1*1步长为1的卷积(1024->256), BN2D, ReLU
                3*3步长为1, padding为2, dilation为2的卷积(256->256), BN2D, ReLU
                1*1步长为1的卷积(256->1024), BN2D, 输出记为out
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 1024, H/8, W/8]
            BottleNeck2 ~ BottleNeck22:
                同BottleNeck1
        layer4: 包含3个Bottoleneck层, 输入为[batch_size, 1024, H/8, W/8]的特征图, 记为x
            BottleNeck0:
                1*1步长为1的卷积(1024->512), BN2D, ReLU
                3*3步长为1, padding为2, dilation为2的卷积(512->512), BN2D, ReLU
                1*1步长为1的卷积(512->2048), BN2D, 输出记为out
                然后将x送入1*1的步长为1的卷积(1024->2048), BN2D中, 输出记为identity
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 2048, H/8, W/8]
            BottleNeck1:
                1*1步长为1的卷积(2048->512), BN2D, ReLU
                3*3步长为1, padding为4, dilation为4的卷积(512->512), BN2D, ReLU
                1*1步长为1的卷积(512->2048), BN2D, 输出记为out
                将out和identity相加, 送入ReLU中, 得到最终的结果, 尺寸为[batch_size, 2048, H/8, W/8]
            BottleNeck2:
                同BottleNeck1
        输出为一个有序字典out:
            out["features"]: layer1的输出, 尺寸为[batch_size, 256, H/4, W/4]
            out["out"]: layer4的输出, 尺寸为[batch_size, 2048, H/8, W/8]
        """
        features = self.model.backbone(x)
        result['features'] = features['features']
        feat_shape = features['features'].shape[-2:]

        # Prediction classification logits
        x = features["out"]
        """
        对CaDDN来说, classifier的输入为[batch_size, 2048, H/8, W/8]记为x
        ASPP:
            x经过一个1*1的步长为1的卷积(2048->256), BN2D, ReLU, 记为res0
            x经过一个3*3的步长为1, padding为12, dilation为12的卷积(2048->256), BN2D, ReLU, 记为res1
            x经过一个3*3的步长为1, padding为24, dilation为24的卷积(2048->256), BN2D, ReLU, 记为res2
            x经过一个3*3的步长为1, padding为36, dilation为36的卷积(2048->256), BN2D, ReLU, 记为res3
            经过一个输出大小为1*1的AdaptiveAvgPool2d, 1*1的步长为1的卷积(2048->256), BN2D, ReLU, 记为res4
                AdaptiveAvgPool2d就是指定输出大小进行avg_pooling, 
                stride = floor ( (input_size / (output_size-1) )
                kernel_size = input_size - (output_size-1) * stride
            res4使用双线性插值进行上采样到尺寸为[batch_size, 256, H/8, W/8], 更新res4
            将res0~res4进行拼接, 得到新的张量res, 尺寸为[batch_size, 1280, H/8, W/8]
        Projection:
            ASPP的输出送入1*1的步长为1的卷积(1280->256), BN2D, ReLU, 最后送入一个概率为0.5的Dropout中
        Projection的结果送入3*3的步长为1, padding为1的卷积(256->256), BN2D, ReLU, 1*1的步长为1的卷积(256->81)
        最后输出的尺寸是[batch_size, 81, H/8, W/8]
 
        """
        x = self.model.classifier(x)
        #* 通过双线性插值将其还原成特征图的大小[batch_size, 81, H/4, W/4]
        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result["logits"] = x

        # Prediction auxillary classification logits
        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for padded pixels
            #* 每个batch中的图像不同尺寸的话会做填充, 这里用0填充
            mask = (x == 0)

            # Match ResNet pretrained preprocessing
            #* 对图像中的信息进行归一化, x = (x-mean)/std
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            #* 对填充的像素也做了归一化, 需要把这些像素重新设置成0
            x[mask] = 0

        return x
