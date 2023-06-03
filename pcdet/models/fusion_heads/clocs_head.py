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

@numba.jit(nopython=True,parallel=True)
def compute_clocs_ious(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
    N = boxes.shape[0] #70400
    K = query_boxes.shape[0] #30
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

    def forward(self, data_dict):
        preds = data_dict['batch_box_preds']
        #* 转成0~1之间
        pred_3d_scores = torch.sigmoid(data_dict['batch_cls_preds']).detach().cpu().numpy()
        #! 计算包围框中心点到lidar的距离, clocs是计算包围框底部中心点到lidar的距离
        dis_to_lidar = torch.norm(preds[:, :, :2],p=2,dim=2,keepdim=True).detach().cpu().numpy()/82.0
        
        calib_V2C_T = data_dict['calib_matrix_V2C_T']
        calib_R0_T = data_dict['calib_matrix_R0_T']
        calib_P2_T = data_dict['calib_matrix_P2_T']
        image_shape = data_dict['image_shape']
        img_height = image_shape[:, 0]
        img_width = image_shape[:, 1]
        
        #* 计算lidar坐标系下的包围框投影到图像上的坐标系
        box_preds_on_image = lidar_boxes_to_image_kitti_torch_cuda(preds, 
                                                                   calib_P2_T, 
                                                                   calib_R0_T,
                                                                   calib_V2C_T, 
                                                                   img_height, 
                                                                   img_width).detach().cpu().numpy()
        
        boxes2d_by_detector = data_dict['results_2d'].detach().cpu().numpy()
        cls_pred_list = []
        valid_flag = []
        for box_2d_preds, box_2d_detector, scores_3d, dist_to_lidar_single in zip(box_preds_on_image, 
                                                                                    boxes2d_by_detector,
                                                                                    pred_3d_scores,
                                                                                    dis_to_lidar):
            cur_pred_2d = box_2d_detector
            k = cur_pred_2d.__len__() - 1
            while k >= 0 and cur_pred_2d[k].sum() == 0:
                k -= 1
            box_2d_detector = cur_pred_2d[:k + 1]
            scores_2d = box_2d_detector[:, 4:5]
            box_2d_detector = box_2d_detector[:, :4]
            overlap = np.zeros((900000,4),dtype=box_2d_preds.dtype)-1
            tensor_index = np.zeros((900000,2),dtype=box_2d_preds.dtype)-1
            ious, tensor_idxs, max_num = compute_clocs_ious(box_2d_preds,
                                                                box_2d_detector,
                                                                -1,
                                                                scores_3d,
                                                                scores_2d,
                                                                dist_to_lidar_single,
                                                                overlap,
                                                                tensor_index)

            ious_tensor = torch.FloatTensor(ious).permute(1,0).reshape(1,4,1,900000)
            tensor_idxs_tensor = torch.LongTensor(tensor_idxs).reshape(-1,2)
            if max_num == 0:
                non_empty_ious_tensor = torch.zeros(1,4,1,2)-1
                non_empty_tensor_idxs_tensor = torch.zeros(2,2)-1
            else:
                non_empty_ious_tensor = ious_tensor[:,:,:,:max_num]
                non_empty_tensor_idxs_tensor = tensor_idxs_tensor[:max_num,:]
            non_empty_tensor_idxs_tensor = non_empty_tensor_idxs_tensor.cuda()
            non_empty_ious_tensor = non_empty_ious_tensor.cuda()
            if non_empty_tensor_idxs_tensor[0,0] == -1:
                output = torch.zeros(1,200,70400,dtype = non_empty_ious_tensor.dtype,device = non_empty_ious_tensor.device)-9999999
                valid_flag.append(False)
            else:
                res = self.fuse(non_empty_ious_tensor)
                output = torch.zeros(1,200,70400,dtype = non_empty_ious_tensor.dtype,device = non_empty_ious_tensor.device)-9999999
                output[:,non_empty_tensor_idxs_tensor[:,0],non_empty_tensor_idxs_tensor[:,1]] = res[0,:,0,:]
                valid_flag.append(True)
                
            output = self.maxpool(output)
            output = output.squeeze().reshape(1,-1,1)
            cls_pred_list.append(output)
        cls_preds = torch.cat(cls_pred_list, dim=0)
        
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['valid_flag'] = valid_flag
        gt_boxes = data_dict['gt_boxes'].detach().cpu().numpy()
        gt_boxes_classes = np.expand_dims(gt_boxes[:, :, -1], axis=-1)
        gt_boxes_in_camera = np.expand_dims(boxes3d_lidar_to_kitti_camera(gt_boxes[0], data_dict['calib'][0]), axis=0)
        gt_boxes_in_camera = np.concatenate([gt_boxes_in_camera, gt_boxes_classes], axis=-1)
        preds_in_lidar = copy.deepcopy(preds)
        #* 在相机坐标系下的预测框
        preds_in_camera = np.expand_dims(boxes3d_lidar_to_kitti_camera(preds[0].detach().cpu().numpy(), data_dict['calib'][0]), axis=0)

        if self.training:
            targets_dict = self.assign_targets(
                preds=preds_in_camera,
                gt_boxes=gt_boxes_in_camera
            )
            self.forward_ret_dict.update(targets_dict)
            
        if not self.training or self.predict_boxes_when_training:
            data_dict['batch_cls_preds'] = cls_preds
            data_dict['batch_box_preds'] = preds_in_lidar
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
            [preds.reshape((1, 200, 176, 1, 2, 7))], gt_boxes
        )
        return targets_dict
