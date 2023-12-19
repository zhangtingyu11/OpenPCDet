import torch.nn as nn
import torch.nn.functional as F

from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class DepthFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            **model_cfg.DDN.ARGS
        )
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        """
        CaDDN
        ddn_result包含两个键值对:
        features: (batch_size, 256, H/4, W/4), 图像特征
        logits: 经过一个分类器之后再上采样到特征图大小, (batch_size, 81, H/4, W/4)
        """
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]

        # Channel reduce
        if self.channel_reduce is not None:
            #* CaDDN中要进行通道降维, 送入1*1的步长为1的卷积(256->64), BN2D, ReLU中
            image_features = self.channel_reduce(image_features)

        # Create image feature plane-sweep volume
        #* 最后输出的尺寸是[batch_size, C, 80(深度的类别数), H/4, W/4], 特征乘上这个类别深度的概率
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)
        batch_dict["frustum_features"] = frustum_features

        if self.training:
            if isinstance(self.ddn_loss, ddn_loss.DDNLoss):
                self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]  #* 深度图的标签
                self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]  #* 2D包围框的标签
                self.forward_ret_dict["depth_logits"] = depth_logits
            elif isinstance(self.ddn_loss, ddn_loss.WeightedDDNLoss):
                self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]  #* 深度图的标签
                self.forward_ret_dict["depth_logits"] = depth_logits
                self.forward_ret_dict["depth_weight"] = batch_dict["depth_weight"].unsqueeze(1)
                
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        #* 对预测的深度值计算softmax, 然后去掉最后一类, 最后一类是超过最大距离的情况
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        #* 最后输出的尺寸是[batch_size, C, 80(深度的类别数), H/4, W/4]
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict
