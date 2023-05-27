import numpy as np
import torch.nn as nn
from ...utils.box_utils import boxes_to_corners_3d
import torch

class ClocsHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class):
        # super().__init__(
        #     model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
        #     predict_boxes_when_training=predict_boxes_when_training
        # )
        super().__init__()

        self.model_cfg = model_cfg
        
        num_filters = self.model_cfg.NUM_FILTERS
        num_filters = [input_channels] + num_filters
        
        self.fuse = []
        for i in range(1, len(num_filters)):
            self.fuse.append(nn.Conv2d(num_filters[i-1], num_filters[i], 1))
            self.fuse.append(nn.ReLU())
        self.fuse = nn.Sequential(*self.fuse)
        self.maxpool_dim = self.model_cfg.MAXPOOL_DIM
        self.maxpool = nn.MaxPool2d([self.model_cfg.MAXPOOL_DIM,1],1),

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.fuse.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.fuse.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        preds = data_dict['batch_box_preds']
        batch_size, pred_num, box_size = preds.shape
        preds = preds.reshape(batch_size*pred_num, box_size)
        pred_corners = boxes_to_corners_3d(preds)
        
        calib = data_dict['calib_matrix']
        pred_corners_homo = torch.cat([pred_corners, torch.ones((*pred_corners.shape[:2], 1)).cuda()], dim=-1)
        pred_corners_homo = pred_corners_homo.view(batch_size, -1, 4)
        
        pred_corners_on_image = torch.einsum('bij,bjk->bik', pred_corners_homo, calib)
        pred_corners_on_image = pred_corners_on_image[:, :, :2]/pred_corners_on_image[:, :, 2].unsqueeze(-1)
        pred_corners_on_image = pred_corners_on_image.view(batch_size, -1, 8, 2)
        
        x_min, _ = torch.min(pred_corners_on_image[:, :, :, 0], dim=-1)
        x_max, _ = torch.max(pred_corners_on_image[:, :, :, 0], dim=-1)
        y_min, _ = torch.min(pred_corners_on_image[:, :, :, 1], dim=-1)
        y_max, _ = torch.max(pred_corners_on_image[:, :, :, 1], dim=-1)
        
        batch_image_shape = data_dict['image_shape']
        img_height = batch_image_shape[:, 0]
        img_width = batch_image_shape[:, 1]
        x_min = torch.clamp(x_min,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1))
        y_min = torch.clamp(y_min,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1))
        x_max = torch.clamp(x_max,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1))
        y_max = torch.clamp(y_max,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1))
        
        
        
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
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
