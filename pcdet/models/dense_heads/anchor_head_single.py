import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        """_summary_

        Args:
            data_dict (_type_): 
                frame_id:帧id
                gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
                points:点云
                flip_x:是否绕着X轴进行翻转
                noise_rot: 整片点云逆时针旋转的角度
                noise_scale: 整片点云缩放的尺度
                use_lead_xyz: 是否使用xyz数据
                voxels: 非空voxel个数(不超过最大voxel个数) * voxel中最大点个数 * 点特征维度
                voxel_coods: voxel的索引, 非空voxel个数 * 4, [zidx, yidx, xidx, batch_idx]
                voxel_num_points: 非空voxel中的点个数(不超过voxel中最大点个数), 非空voxel个数
                image_shape: 图像尺寸
                batch_size: batch_size
                pillar_features: 非空voxel个数 * 特征维度的voxel特征
                spatial_features: BEV特征图, batch_size * 特征维度 * H * W
                spatial_features_2d: 骨干网络输出的二维特征图, batch_size * 特征维度 * H * W
        Returns:
            _type_: _description_
        """
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            """
                self.forward_ret_dict
                    cls_preds: 预测的类别, [batch_size, H, W, num_class * num_class * anchor_num]
                    box_preds:  预测的regression, [batch_size, H, W, 7 * num_class * anchor_num]
                    dir_cls_preds: 预测的航向角, [batch_size, H, W, num_class * anchor_num * angle_bin]
                    box_cls_labels: [batch_size, 所有类别的anchor个数和], 0表示背景, >0表示前景, -1表示忽略
                    box_reg_targets: [batch_size, 所有类别的anchor个数和, 7], 只对前景求regression, 其他的都是0
                    reg_weights: [batch_size, 所有类别的anchor个数和], 前景点的weight是1
            """
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        """
            返回的data_dict (_type_): 
                frame_id:帧id
                gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
                points:点云
                flip_x:是否绕着X轴进行翻转
                noise_rot: 整片点云逆时针旋转的角度
                noise_scale: 整片点云缩放的尺度
                use_lead_xyz: 是否使用xyz数据
                voxels: 非空voxel个数(不超过最大voxel个数) * voxel中最大点个数 * 点特征维度
                voxel_coods: voxel的索引, 非空voxel个数 * 4, [zidx, yidx, xidx, batch_idx]
                voxel_num_points: 非空voxel中的点个数(不超过voxel中最大点个数), 非空voxel个数
                image_shape: 图像尺寸
                batch_size: batch_size
                pillar_features: 非空voxel个数 * 特征维度的voxel特征
                spatial_features: BEV特征图, batch_size * 特征维度 * H * W
                spatial_features_2d: 骨干网络输出的二维特征图, batch_size * 特征维度 * H * W
        """
        return data_dict
