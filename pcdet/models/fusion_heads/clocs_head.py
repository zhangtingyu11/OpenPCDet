import numpy as np
import torch.nn as nn
from ...utils.box_utils import boxes_to_corners_3d, pairwise_iou
from ...utils import box_utils, common_utils, calibration_kitti
from ...ops.clocs.clocs_utils import compute_clocs_iou
from mmdet.models.task_modules.assigners import BboxOverlaps2D
from ..dense_heads.anchor_head_template import AnchorHeadTemplate
import torch
from numba import njit, jit

@jit(parallel=True, forceobj=True)
def build_stage2_training(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
    N = boxes.shape[0] #70400
    K = query_boxes.shape[0] #30
    max_num = 900000
    ind=0
    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[ind,0] = iw * ih / ua
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = scores_2d[k,0]
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    ind = ind+1

                elif k==K-1:
                    overlaps[ind,0] = -10
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    ind = ind+1
            elif k==K-1:
                overlaps[ind,0] = -10
                overlaps[ind,1] = scores_3d[n,0]
                overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]
                tensor_index[ind,0] = k
                tensor_index[ind,1] = n
                ind = ind+1
    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind
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

        self.compute_ious = BboxOverlaps2D()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.fuse.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.fuse.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        preds = data_dict['batch_box_preds']
        pred_scores = torch.sigmoid(data_dict['batch_cls_preds'])
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
        
        anchor_project_on_image = torch.cat([x_min, y_min, x_max, y_max], dim=-1)
        
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
        preds_2d_project_on_image = data_dict['results_2d']
        cls_pred_list = []
        valid_flag = []
        # for bidx, (anchor_projected, gt_projected) in enumerate(zip(anchor_project_on_image, gt_project_on_image)):
        #     cur_iou = self.compute_ious(anchor_projected, gt_projected)
        #     nonzero_indexes = cur_iou.nonzero()
        #     if(nonzero_indexes.shape[0] == 0):
        #         out_1 = torch.zeros(1,200,70400,dtype = anchor_projected.dtype,device = anchor_projected.device)-9999999
        #     else:
        #         overlaps = torch.zeros((nonzero_indexes.shape[0], 4), dtype = anchor_projected.dtype, device = anchor_projected.device)
        #         overlaps[:, 0] = cur_iou[nonzero_indexes[:, 0], nonzero_indexes[:, 1]]
        #         overlaps[:, 1] = pred_scores[bidx][nonzero_indexes[:, 0]].squeeze(-1)
        #         overlaps[:, 2] = gt_project_on_image[bidx][nonzero_indexes[:, 1], 4]
        #         overlaps[:, 3] = dis_to_lidar[bidx][nonzero_indexes[:, 0], 0]
        #         overlaps = overlaps.transpose(0,1).unsqueeze(1).unsqueeze(0)
        #         output = self.fuse(overlaps)
        #         out_1 = torch.zeros(1,200, 70400, dtype=output.dtype, device = output.device)-9999999
        #         out_1[:,nonzero_indexes[:,1],nonzero_indexes[:,0]] = output[0,:,0,:]
        #     output = self.maxpool(out_1)
        #     cls_preds = output.reshape(1, *data_dict['spatial_features_2d'].shape[-2:], 2)
        #     cls_pred_list.append(cls_preds)
        
        for bidx, (box_2d_preds, box_2d_detector) in enumerate(zip(anchor_project_on_image, preds_2d_project_on_image)):
            cur_pred_2d = box_2d_detector
            k = cur_pred_2d.__len__() - 1
            while k >= 0 and cur_pred_2d[k].sum() == 0:
                k -= 1
            box_2d_detector = cur_pred_2d[:k + 1]
            overlaps1 = torch.zeros((900000,4),dtype=box_2d_preds.dtype, device = box_2d_preds.device)
            tensor_index1 = torch.zeros((900000,2),dtype=torch.int, device = box_2d_preds.device)
            overlaps1[:,:] = -1
            tensor_index1[:,:] = -1
            #final_scores[final_scores<0.1] = 0
            #box_2d_preds[(final_scores<0.1).reshape(-1),:] = 0 
            max_num = torch.zeros(1, dtype=torch.int, device = box_2d_preds.device)
            iou_test,tensor_index, max_num = compute_clocs_iou(box_2d_preds,
                                                box_2d_detector[:, :4].contiguous(),
                                                pred_scores[bidx],
                                                box_2d_detector[:, -1].unsqueeze(-1).contiguous(),
                                                dis_to_lidar[bidx],
                                                overlaps1,
                                                tensor_index1,
                                                max_num)
            iou_test_tensor = iou_test  #iou_test_tensor shape: [160000,4]
            tensor_index_tensor = tensor_index
            iou_test_tensor = iou_test_tensor.permute(1,0)
            iou_test_tensor = iou_test_tensor.reshape(1,4,1,900000)
            tensor_index_tensor = tensor_index_tensor.reshape(-1,2)
            max_num = max_num.item()
            if max_num == 0:
                non_empty_iou_test_tensor = torch.zeros(1,4,1,2)
                non_empty_iou_test_tensor[:,:,:,:] = -1
                non_empty_tensor_index_tensor = torch.zeros(2,2)
                non_empty_tensor_index_tensor[:,:] = -1
            else:
                non_empty_iou_test_tensor = iou_test_tensor[:,:,:,:max_num]
                non_empty_tensor_index_tensor = tensor_index_tensor[:max_num,:]
            if non_empty_tensor_index_tensor[0,0] == -1:
                out_1 = torch.zeros(1,200,70400,dtype = non_empty_iou_test_tensor.dtype,device = non_empty_iou_test_tensor.device)
                out_1[:,:,:] = -9999999
                valid_flag.append(0)
            else:
                x = self.fuse(non_empty_iou_test_tensor)
                out_1 = torch.zeros(1,200,70400,dtype = non_empty_iou_test_tensor.dtype,device = non_empty_iou_test_tensor.device)
                out_1[:,:,:] = -9999999
                out_1[:,non_empty_tensor_index_tensor[:,0],non_empty_tensor_index_tensor[:,1]] = x[0,:,0,:]
                valid_flag.append(1)
                
            x = self.maxpool(out_1)
            #x, _ = torch.max(out_1,1)
            x = x.squeeze().reshape(1,-1,1)
            cls_pred_list.append(x)
        cls_preds = torch.cat(cls_pred_list, dim=0)
        
        
        
        
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['valid_flag'] = valid_flag
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

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
