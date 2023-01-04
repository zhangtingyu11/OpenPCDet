import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                frame_id:帧id
                gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
                points:点云
                flip_x:是否绕着X轴进行翻转
                noise_rot: 整片点云逆时针旋转的角度
                noise_scale: 整片点云缩放的尺度
                use_lead_xyz: 是否使用xyz数据
                image_shape: 图像尺寸
                batch_size: batch_size
                points_features: (B*N, C)的点特征
                points_coords: (B*N, 4)的batch_idx + 点坐标
                point_cls_scores: (B*N), 点的分类分数(取最高的那个)
                batch_cls_preds: (B*N, 3), 点的预测结果
                batch_box_preds: (B*N, 7), 包围框的预测结果, [x, y, z, dx, dy, dz, heading]
                cls_preds_normalized: 
                roi: (batch_size, roi个数, 7)的roi包围框
                roi_scores: (batch_size, roi个数)的roi分数
                roi_labels: (batch_size, roi个数)的类别标签
                has_class_labels: True表示roi_labels里面按照类别分不同的标签
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        """
            batch_rois: roi包围框, (batch_size, ROI_PER_IMAGE, box_code_size)
            batch_gt_of_rois: roi对应的gt框, (batch_size, ROI_PER_IMAGE, box_code_size+1), 最后一维是gt框的类别
            batch_roi_ious: roi和其对应的gt的iou, (batch_size, ROI_PER_IMAGE)
            batch_roi_scores: roi的置信度, (batch_size, ROI_PER_IMAGE)
            batch_roi_labels: roi对应的分类标签, (batch_size, ROI_PER_IMAGE)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # regression valid mask
        #* 只对iou大于REG_FG_THRESH的roi进行回归
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            #* iou > CLS_FG_THRESH则标签为1, iou < CLS_BG_THRESH则标签为0, 在CLS_BG_THRESH~CLS_FG_THRESH之间则标签为-1
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        """
        Returns:
            rois: roi包围框, (batch_size, ROI_PER_IMAGE, box_code_size)
            gt_of_rois: roi对应的gt框, (batch_size, ROI_PER_IMAGE, box_code_size+1), 最后一维是gt框的类别
            gt_iou_of_rois: roi和其对应的gt的iou, (batch_size, ROI_PER_IMAGE)
            roi_scores: roi的置信度, (batch_size, ROI_PER_IMAGE)
            roi_labels: roi对应的分类标签, (batch_size, ROI_PER_IMAGE)
            reg_valid_mask: roi是否需要计算regression loss的掩码, (batch_size, ROI_PER_IMAGE)
            rcnn_cls_labels: roi的前背景分类, 1为前景, 0为背景, -1为忽略
        """
        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                frame_id:帧id
                gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
                points:点云
                flip_x:是否绕着X轴进行翻转
                noise_rot: 整片点云逆时针旋转的角度
                noise_scale: 整片点云缩放的尺度
                use_lead_xyz: 是否使用xyz数据
                image_shape: 图像尺寸
                batch_size: batch_size
                points_features: (B*N, C)的点特征
                points_coords: (B*N, 4)的batch_idx + 点坐标
                point_cls_scores: (B*N), 点的分类分数(取最高的那个)
                batch_cls_preds: (B*N, 3), 点的预测结果
                batch_box_preds: (B*N, 7), 包围框的预测结果, [x, y, z, dx, dy, dz, heading]
                cls_preds_normalized: 
                roi: (batch_size, roi个数, 7)的roi包围框
                roi_scores: (batch_size, roi个数)的roi分数
                roi_labels: (batch_size, roi个数)的类别标签
                has_class_labels: True表示roi_labels里面按照类别分不同的标签
        Returns:
        
        """
        batch_size = batch_dict['batch_size']
        #* (batch_size, roi个数, 7)的roi包围框
        rois = batch_dict['rois']
        #* (batch_size, roi个数)的roi分数
        roi_scores = batch_dict['roi_scores']
        #* (batch_size, roi个数)的类别标签
        roi_labels = batch_dict['roi_labels']
        #* (batch_size, gt个数, 8)的gt包围框, [x, y, z, dx, dy, dz, heading, class]
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]
        #* roi包围框, (batch_size, ROI_PER_IMAGE, box_code_size)
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        #* roi对应的gt框, (batch_size, ROI_PER_IMAGE, box_code_size+1), 最后一维是gt框的类别
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        #* roi和其对应的gt的iou, (batch_size, ROI_PER_IMAGE)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        #* roi的置信度, (batch_size, ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        #* roi对应的分类标签, (batch_size, ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            #* roi按照fg, easy_bg, hard_bg一定比例进行采样
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
        """
        Returns:
            batch_rois: roi包围框, (batch_size, ROI_PER_IMAGE, box_code_size)
            batch_gt_of_rois: roi对应的gt框, (batch_size, ROI_PER_IMAGE, box_code_size+1), 最后一维是gt框的类别
            batch_roi_ious: roi和其对应的gt的iou, (batch_size, ROI_PER_IMAGE)
            batch_roi_scores: roi的置信度, (batch_size, ROI_PER_IMAGE)
            batch_roi_labels: roi对应的分类标签, (batch_size, ROI_PER_IMAGE)
        """
        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        #* 前景框需要采样的个数
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        #* REG_FG_THRESH和CLS_FG_THRESH取最小作为前景框阈值
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        #* 简单易分类的背景框: iou小于CLS_BG_THRESH_LO
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        #* 难于分类的背景框: iou小于REG_FG_THRESH， iou大于等于CLS_BG_THRESH_LO
        #* 一般来说REG_FG_THRESH应该小于CLS_FG_THRESH, 不然会出现有样本既是fg又是hard_bg_inds
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            #* 优先采样前景框
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            #* 前景框采样完采样背景框
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        #* 优先保证hard_bg
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:
            
        Returns:
        
        """
        """
        :param max_overlaps: roi和gt的最大iou值, [roi个数]
        :param gt_assignment: 和roi iou最大的gt框的索引
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:, :7], cur_gt[:, :7])  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
