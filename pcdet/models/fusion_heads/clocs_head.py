import numpy as np
import torch.nn as nn
from ...utils.box_utils import boxes_to_corners_3d, pairwise_iou
from ...utils import box_utils, common_utils, calibration_kitti
from mmdet.models.task_modules.assigners import BboxOverlaps2D

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
        self.compute_ious = BboxOverlaps2D()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.fuse.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.fuse.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        preds = data_dict['batch_box_preds']
        pred_scores = data_dict['batch_cls_preds']
        dis_to_lidar = torch.norm(preds[:, :, :2],p=2,dim=2,keepdim=True)/82.0
        batch_size, pred_num, box_size = preds.shape
        preds = preds.view(batch_size*pred_num, box_size)
        pred_corners = boxes_to_corners_3d(preds)
        
        calib_V2C_T = data_dict['calib_matrix_V2C_T']
        calib_R0_T = data_dict['calib_matrix_R0_T']
        calib_P2_T = data_dict['calib_matrix_P2_T']
        
        pred_corners_homo = torch.cat([pred_corners, torch.ones((*pred_corners.shape[:2], 1)).cuda()], dim=-1)
        pred_corners_homo = pred_corners_homo.view(batch_size, -1, 4)
        lidar_to_rect_matrix = torch.einsum('bij, bjk->bik', calib_V2C_T, calib_R0_T)
        pred_corners_rect = torch.einsum('bij, bjk->bik', pred_corners_homo, lidar_to_rect_matrix)
        
        pred_corners_rect_homo = torch.cat([pred_corners_rect, torch.ones((*pred_corners_rect.shape[:2], 1)).cuda()], dim=-1)
        
        pred_corners_on_image = torch.einsum('bij,bjk->bik', pred_corners_rect_homo, calib_P2_T).view(-1, 3)
        pred_corners_on_image = (pred_corners_on_image[:, :2].transpose(0,1)/pred_corners_on_image[:, 2].unsqueeze(0)).transpose(0,1)
        pred_corners_on_image = pred_corners_on_image.view(batch_size, -1, 8, 2)
        
        x_min, _ = torch.min(pred_corners_on_image[:, :, :, 0], dim=-1)
        x_max, _ = torch.max(pred_corners_on_image[:, :, :, 0], dim=-1)
        y_min, _ = torch.min(pred_corners_on_image[:, :, :, 1], dim=-1)
        y_max, _ = torch.max(pred_corners_on_image[:, :, :, 1], dim=-1)
        
        batch_image_shape = data_dict['image_shape']
        img_height = batch_image_shape[:, 0]
        img_width = batch_image_shape[:, 1]
        x_min = torch.clamp(x_min,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1)).unsqueeze(-1)
        y_min = torch.clamp(y_min,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1)).unsqueeze(-1)
        x_max = torch.clamp(x_max,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1)).unsqueeze(-1)
        y_max = torch.clamp(y_max,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1)).unsqueeze(-1)
        
        anchor_project_on_image = torch.cat([x_min, y_min, x_max, y_max, pred_scores], dim=-1)
        
        #TODO 先用gt 2D框试试看，CLOCs用的是预测框
        gt_corners = boxes_to_corners_3d(data_dict['gt_boxes'][:, :, :7].reshape(-1, 7).cpu().numpy())
        gt_corners = torch.from_numpy(gt_corners).cuda()
        gt_corners_homo = torch.cat([gt_corners, torch.ones((*gt_corners.shape[:2], 1)).cuda()], dim=-1)
        gt_corners_homo = gt_corners_homo.view(batch_size, -1, 4)
        
        gt_corners_rect = torch.einsum('bij,bjk->bik', gt_corners_homo, lidar_to_rect_matrix)
        
        gt_corners_rect_homo = torch.cat([gt_corners_rect, torch.ones((*gt_corners_rect.shape[:2], 1)).cuda()], dim=-1)
        
        gt_corners_on_image = torch.einsum('bij,bjk->bik', gt_corners_rect_homo, calib_P2_T).view(-1, 3)
        gt_corners_on_image = (gt_corners_on_image[:, :2].transpose(0,1)/gt_corners_on_image[:, 2].unsqueeze(0)).transpose(0,1)
        gt_corners_on_image = gt_corners_on_image.view(batch_size, -1, 8, 2)
        
        x_min, _ = torch.min(gt_corners_on_image[:, :, :, 0], dim=-1)
        x_max, _ = torch.max(gt_corners_on_image[:, :, :, 0], dim=-1)
        y_min, _ = torch.min(gt_corners_on_image[:, :, :, 1], dim=-1)
        y_max, _ = torch.max(gt_corners_on_image[:, :, :, 1], dim=-1)
        
        x_min = torch.clamp(x_min,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1)).unsqueeze(-1)
        y_min = torch.clamp(y_min,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1)).unsqueeze(-1)
        x_max = torch.clamp(x_max,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1)).unsqueeze(-1)
        y_max = torch.clamp(y_max,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1)).unsqueeze(-1)
        
        gt_project_on_image = torch.cat([x_min, y_min, x_max, y_max, torch.ones_like(x_min)], dim=-1)
        
        overlaps1 = torch.zeros((batch_size, 900000, 4), dtype=gt_project_on_image.dtype)-1
        tensor_index1 = torch.zeros((batch_size, 900000, 2), dtype=gt_project_on_image.dtype)-1
        for bidx, (anchor_projected, gt_projected) in enumerate(zip(anchor_project_on_image, gt_project_on_image)):
            cur_iou = self.compute_ious(anchor_projected, gt_projected)
            nonzero_indexes = cur_iou.nonzero()
            overlaps1[bidx][:nonzero_indexes.shape[0], 0] = cur_iou[nonzero_indexes[:, 0], nonzero_indexes[:, 1]]
            overlaps1[bidx][:nonzero_indexes.shape[0], 1] = torch.sigmoid(pred_scores[bidx][nonzero_indexes[:, 0]].squeeze(-1))
            overlaps1[bidx][:nonzero_indexes.shape[0], 2] = gt_project_on_image[bidx][nonzero_indexes[:, 1], 4]
            overlaps1[bidx][:nonzero_indexes.shape[0], 3] = dis_to_lidar[bidx][nonzero_indexes[:, 0], 0]
            tensor_index1[bidx][:nonzero_indexes.shape[0], 0] = nonzero_indexes[:, 0]
            tensor_index1[bidx][:nonzero_indexes.shape[0], 1] = nonzero_indexes[:, 1]
        return data_dict
