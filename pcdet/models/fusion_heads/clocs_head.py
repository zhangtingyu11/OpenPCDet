import numpy as np
import torch.nn as nn
from ...utils.box_utils import boxes3d_lidar_to_kitti_camera, lidar_boxes_to_image_kitti_torch_cuda
from ...utils import box_utils, common_utils, calibration_kitti
from ...ops.clocs.clocs_utils import compute_clocs_iou
from ..dense_heads.anchor_head_template import AnchorHeadTemplate
import torch
import numba
import copy

# def lidar_to_camera(points, r_rect, velo2cam):
#     num_points = points.shape[0]
#     points = torch.cat(
#         [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
#     camera_points = points @ (r_rect @ velo2cam).t()
#     return camera_points[..., :3]

# def box_lidar_to_camera(data, r_rect, velo2cam):
#     xyz_lidar = data[..., 0:3]
#     w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
#     r = data[..., 6:7]
#     xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
#     return torch.cat([xyz, l, h, w, r], dim=-1)

# @numba.jit(nopython=True,parallel=True)
# def build_stage2_training(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
#     N = boxes.shape[0] #70400
#     K = query_boxes.shape[0] #30
#     ind=0
#     ind_max = ind
#     for k in range(K):
#         qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
#                      (query_boxes[k, 3] - query_boxes[k, 1]))
#         for n in range(N):
#             iw = (min(boxes[n, 2], query_boxes[k, 2]) -
#                   max(boxes[n, 0], query_boxes[k, 0]))
#             if iw > 0:
#                 ih = (min(boxes[n, 3], query_boxes[k, 3]) -
#                       max(boxes[n, 1], query_boxes[k, 1]))
#                 if ih > 0:
#                     if criterion == -1:
#                         ua = (
#                             (boxes[n, 2] - boxes[n, 0]) *
#                             (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
#                     elif criterion == 0:
#                         ua = ((boxes[n, 2] - boxes[n, 0]) *
#                               (boxes[n, 3] - boxes[n, 1]))
#                     elif criterion == 1:
#                         ua = qbox_area
#                     else:
#                         ua = 1.0
#                     overlaps[ind,0] = iw * ih / ua
#                     overlaps[ind,1] = scores_3d[n,0]
#                     overlaps[ind,2] = scores_2d[k,0]
#                     overlaps[ind,3] = dis_to_lidar_3d[n,0]
#                     tensor_index[ind,0] = k
#                     tensor_index[ind,1] = n
#                     ind = ind+1

#                 elif k==K-1:
#                     overlaps[ind,0] = -10
#                     overlaps[ind,1] = scores_3d[n,0]
#                     overlaps[ind,2] = -10
#                     overlaps[ind,3] = dis_to_lidar_3d[n,0]
#                     tensor_index[ind,0] = k
#                     tensor_index[ind,1] = n
#                     ind = ind+1
#             elif k==K-1:
#                 overlaps[ind,0] = -10
#                 overlaps[ind,1] = scores_3d[n,0]
#                 overlaps[ind,2] = -10
#                 overlaps[ind,3] = dis_to_lidar_3d[n,0]
#                 tensor_index[ind,0] = k
#                 tensor_index[ind,1] = n
#                 ind = ind+1
#     if ind > ind_max:
#         ind_max = ind
#     return overlaps, tensor_index, ind
class ClocsHead(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.model_cfg = model_cfg
        
        num_filters = self.model_cfg.NUM_FILTERS
        num_filters = [input_channels] + num_filters
        
        self.fuse = []
        for i in range(1, len(num_filters)-1):
            self.fuse.append(nn.Conv2d(num_filters[i-1], num_filters[i], 1))
            self.fuse.append(nn.ReLU())
        self.fuse.append(nn.Conv2d(num_filters[-2], num_filters[-1], 1))
        self.fuse = nn.Sequential(*self.fuse)
        self.maxpool_dim = self.model_cfg.MAXPOOL_DIM
        self.maxpool = nn.MaxPool2d([self.model_cfg.MAXPOOL_DIM,1],1)
        # self.init_weights()

    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.fuse.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.fuse.weight, mean=0, std=0.001)
        # pass

    def forward(self, data_dict):
        preds = data_dict['batch_box_preds']
        #* 转成0~1之间
        pred_scores_3d = torch.sigmoid(data_dict['batch_cls_preds'])
        #! 计算包围框中心点到lidar的距离, clocs是计算包围框底部中心点到lidar的距离
        dis_to_lidar = torch.norm(preds[:, :, :2],p=2,dim=2,keepdim=True)/82.0
        
        calib_V2C_T = data_dict['calib_matrix_V2C_T']
        calib_R0_T = data_dict['calib_matrix_R0_T']
        calib_P2_T = data_dict['calib_matrix_P2_T']
        image_shape = data_dict['image_shape']
        img_height = image_shape[:, 0]
        img_width = image_shape[:, 1]
        
        box_preds_on_image = lidar_boxes_to_image_kitti_torch_cuda(preds, calib_P2_T, calib_R0_T,
                                                                   calib_V2C_T, img_height, img_width)
        
        boxes2d_by_detector = data_dict['results_2d']
        
        #! 注释的是用gt 2D框代替预测框
        # gt_corners = boxes_to_corners_3d(data_dict['gt_boxes'][:, :, :7].reshape(-1, 7).cpu().numpy())
        # gt_corners = torch.from_numpy(gt_corners).cuda()
        # gt_corners_homo = torch.cat([gt_corners, torch.ones((*gt_corners.shape[:2], 1)).cuda()], dim=-1)
        # gt_corners_homo = gt_corners_homo.view(batch_size, -1, 4)
        
        # gt_corners_rect = torch.einsum('bij,bjk->bik', gt_corners_homo, lidar_to_rect_matrix)
        
        # gt_corners_rect_homo = torch.cat([gt_corners_rect, torch.ones((*gt_corners_rect.shape[:2], 1)).cuda()], dim=-1)
        
        # gt_corners_on_image = torch.einsum('bij,bjk->bik', gt_corners_rect_homo, calib_P2_T).view(-1, 3)
        # gt_corners_on_image = (gt_corners_on_image[:, :2].transpose(0,1)/gt_corners_on_image[:, 2].unsqueeze(0)).transpose(0,1)
        # gt_corners_on_image = gt_corners_on_image.view(batch_size, -1, 8, 2)
        
        # x_min, _ = torch.min(gt_corners_on_image[:, :, :, 0], dim=-1)
        # x_max, _ = torch.max(gt_corners_on_image[:, :, :, 0], dim=-1)
        # y_min, _ = torch.min(gt_corners_on_image[:, :, :, 1], dim=-1)
        # y_max, _ = torch.max(gt_corners_on_image[:, :, :, 1], dim=-1)
        
        # x_min = torch.clamp(x_min,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1)).unsqueeze(-1)
        # y_min = torch.clamp(y_min,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1)).unsqueeze(-1)
        # x_max = torch.clamp(x_max,min = torch.zeros(batch_size, 1).cuda(),max = img_width.unsqueeze(-1)).unsqueeze(-1)
        # y_max = torch.clamp(y_max,min = torch.zeros(batch_size, 1).cuda(),max = img_height.unsqueeze(-1)).unsqueeze(-1)
        
        # gt_project_on_image = torch.cat([x_min, y_min, x_max, y_max, torch.ones_like(x_min)], dim=-1)
        # anchor_project_on_image = data_dict['results_3d'][:, :, :4].cpu().numpy()
        # dis_to_lidar = data_dict['dis_to_lidar'][0].cpu().numpy()
        # final_scores = data_dict['results_3d'][:, :, 4].contiguous().view(-1 , 1).cpu().numpy()
        # preds_2d_project_on_image = data_dict['results_2d'].cpu().numpy()
        cls_pred_list = []
        valid_flag = []
        for (anchor_projected, boxes2d_projected, pred_score_3d, dis_to_lidar_single) in zip(box_preds_on_image, 
                                                                        boxes2d_by_detector, 
                                                                        pred_scores_3d,
                                                                        dis_to_lidar):
            boxes2d_projected_scores = boxes2d_projected[:, -1]
            boxes2d_projected = boxes2d_projected[:, :4]
            boxes_3d_num = anchor_projected.shape[0]
            boxes_2d_num = boxes2d_projected.shape[0]
            overlap = torch.zeros((boxes_3d_num, boxes_2d_num, 4), 
                                  dtype = torch.float32,
                                  device = anchor_projected.device)-1
            tensor_index = torch.zeros((boxes_3d_num, boxes_2d_num, 2), 
                                    dtype = torch.int,
                                    device = anchor_projected.device)-1
            overlap, tensor_index = compute_clocs_iou(anchor_projected,
                                          boxes2d_projected.contiguous(),
                                          pred_score_3d.squeeze(-1),
                                          boxes2d_projected_scores.contiguous(),
                                          dis_to_lidar_single.squeeze(-1),
                                          overlap,
                                          tensor_index)
            tensor_index = torch.sum(tensor_index, dim=-1) != -2
            nonzero_indexes = tensor_index.nonzero()
            torch.sort(nonzero_indexes)
            if(nonzero_indexes.shape[0] == 0):
                non_empty_iou_test_tensor = torch.zeros(1,4,1,2)
                non_empty_iou_test_tensor[:,:,:,:] = -1
                non_empty_tensor_index_tensor = torch.zeros(2,2)
                non_empty_tensor_index_tensor[:,:] = -1
                output = torch.zeros(1,200,70400,dtype = non_empty_iou_test_tensor.dtype,device = non_empty_iou_test_tensor.device)-9999999
                valid_flag.append(False)
            else:
                non_empty_iou_test_tensor = overlap[nonzero_indexes[:, 0], nonzero_indexes[:, 1]]
                non_empty_iou_test_tensor = non_empty_iou_test_tensor.transpose(0, 1)[None, :, None, :]
                non_empty_tensor_index_tensor = tensor_index[nonzero_indexes[:, 0], nonzero_indexes[:, 1]]
                output = torch.zeros(1,200, 70400, dtype=non_empty_iou_test_tensor.dtype, device = non_empty_iou_test_tensor.device)-9999999
                res = self.fuse(non_empty_iou_test_tensor)
                output[:,nonzero_indexes[:,1],nonzero_indexes[:,0]] = res[0,:,0,:]
                valid_flag.append(True)
            output = self.maxpool(output)
            cls_preds = output.reshape(1, 70400, 1)
            cls_pred_list.append(cls_preds)

        
        batch_cls_preds = torch.cat(cls_pred_list, dim=0)
        self.forward_ret_dict['cls_preds'] =batch_cls_preds
        self.forward_ret_dict['valid_flag'] =valid_flag

        gt_boxes = data_dict['gt_boxes']
        gt_boxes_class = gt_boxes[:, :, 7:8]
        gt_boxes = np.expand_dims(boxes3d_lidar_to_kitti_camera(gt_boxes[0].detach().cpu().numpy(), data_dict['calib'][0]), axis=0)
        pred_boxes = np.expand_dims(boxes3d_lidar_to_kitti_camera(preds[0].detach().cpu().numpy(), data_dict['calib'][0]), axis=0)
        gt_boxes = torch.from_numpy(gt_boxes).cuda()
        gt_boxes = torch.cat([gt_boxes, gt_boxes_class], dim=-1)
        pred_boxes = torch.from_numpy(pred_boxes).cuda()
        if self.training:
            targets_dict = self.assign_targets(
                preds=pred_boxes,
                gt_boxes=gt_boxes
            )
            self.forward_ret_dict.update(targets_dict)
            
        if not self.training or self.predict_boxes_when_training:
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = preds
            data_dict['cls_preds_normalized'] = False


        return data_dict
    
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        fusion_loss = cls_loss
        tb_dict['fusion_loss'] = fusion_loss.item()
        return fusion_loss, tb_dict
    
    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        for idx, valid in enumerate(self.forward_ret_dict['valid_flag']):
            if(valid == 0):
                box_cls_labels[idx]=-1
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict
    
    def assign_targets(self, preds, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            [preds.view((1, 200, 176, 1, 2, 7))], gt_boxes
        )
        return targets_dict
    
    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        return cls_preds, box_preds