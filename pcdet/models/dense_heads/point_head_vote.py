import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...utils import box_coder_utils, box_utils, common_utils, loss_utils
from .point_head_template import PointHeadTemplate



class PointHeadVote(PointHeadTemplate):
    """
    A simple vote-based detection head, which is used for 3DSSD.
    Reference Paper: https://arxiv.org/abs/2002.10187
    3DSSD: Point-based 3D Single Stage Object Detector
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        use_bn = self.model_cfg.USE_BN
        self.predict_boxes_when_training = predict_boxes_when_training

        self.vote_cfg = self.model_cfg.VOTE_CONFIG
        self.vote_layers = self.make_fc_layers(
            input_channels=input_channels,
            output_channels=3,
            fc_list=self.vote_cfg.VOTE_FC
        )

        self.sa_cfg = self.model_cfg.SA_CONFIG
        channel_in, channel_out = input_channels, 0
        
        self.only_cls_confident_points = self.model_cfg.TARGET_CONFIG.get("ONLY_CLS_CONFIDENT_POINTS", False)
        self.confident_points_per_batch = self.model_cfg.TARGET_CONFIG.get("CONFIDENT_POINTS_PER_BATCH", 100)
        self.confident_points_flag = False


        mlps = self.sa_cfg.MLPS.copy()
        for idx in range(mlps.__len__()):
            mlps[idx] = [channel_in] + mlps[idx]
            channel_out += mlps[idx][-1]
            
        use_density = self.model_cfg.SA_CONFIG.get("USE_DENSITY", False)
        use_kde = self.model_cfg.SA_CONFIG.get("USE_KDE", False)
        use_distance_to_center = self.model_cfg.SA_CONFIG.get("USE_DISTANCE_TO_CENTER", False)
        use_relative_direction_angle = self.model_cfg.SA_CONFIG.get("USE_RELATIVE_DIRECTION_ANGLE", False)
        
        self.SA_module = pointnet2_modules.PointnetSAModuleFSMSG(
            radii=self.sa_cfg.RADIUS,
            nsamples=self.sa_cfg.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            bn=use_bn,
            use_density = use_density,
            use_kde = use_kde,
            use_distance_to_center = use_distance_to_center,
            use_relative_direction_angle = use_relative_direction_angle,
        )

        channel_in = channel_out
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(channel_in, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            channel_in = self.model_cfg.SHARED_FC[k]

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        channel_in = self.model_cfg.SHARED_FC[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=num_class if not self.model_cfg.LOSS_CONFIG.LOSS_CLS == 'CrossEntropy' else num_class + 1,
            fc_list=self.model_cfg.CLS_FC
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size,
            fc_list=self.model_cfg.REG_FC
        )

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):
        # classification loss
        if losses_cfg.LOSS_CLS == 'WeightedBinaryCrossEntropy':
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedBinaryCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS == 'WeightedCrossEntropy':
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS == 'FocalLoss':
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_CLS.startswith('WeightedSmoothL1Loss'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=None,
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_CLS.startswith('WeightedL1Loss'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=None
                )
            )
        else:
            raise NotImplementedError

        # regression loss
        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError

        # sasa loss
        loss_sasa_cfg = losses_cfg.get('LOSS_SASA_CONFIG', None)
        if loss_sasa_cfg is not None:
            self.enable_sasa = True
            self.add_module(
                'loss_point_sasa',
                loss_utils.PointSASALoss(**loss_sasa_cfg)
            )
        else:
            self.enable_sasa = False

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        """构造全连接层
        1dconv -> bn -> relu
        ...
        1dconv -> bn -> relu
        1dconv

        Args:
            input_channels (_type_): 输入通道
            output_channels (_type_): 输出通道
            fc_list (_type_): 全连接层的隐藏通道数

        Returns:
            _type_: 全连接层模块
        """
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def assign_stack_targets_simple(self, points, gt_boxes, extend_gt_boxes=None, set_ignore_flag=True):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: (B, M, 8), required if set ignore flag
            set_ignore_flag:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignore
            point_reg_labels: (N1 + N2 + N3 + ..., 3), corresponding object centroid
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert not set_ignore_flag or extend_gt_boxes is not None
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], 3))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            #* point_cls_labels_single是记录batch里一个sample的点的分类标签, [一个sample中的点个数]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            #* box_idxs_of_pts用来存储每个点所在的包围框的索引, 背景点默认是-1, [一个sample中的点个数]
            #* 看这个点是否在gt包围框，而不是在扩大后的包围框
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            #* 前景点的flag
            box_fg_flag = (box_idxs_of_pts >= 0)

            if extend_gt_boxes is not None:
                #* extend_box_idx_of_pts用来存储每个点所在的扩大后的包围框的索引, 背景点默认是-1, [一个sample中的点个数]
                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                #* 如果不在原始gt框， 在扩大后的gt框， 那ignore_flag就是1
                ignore_flag = fg_flag ^ (extend_box_idx_of_pts >= 0)
                #* 那么不在原始gt框，在扩大后的gt框的点的类别标签是-1， 表示被忽略
                #* 这样的话背景点类别标签就是0
                point_cls_labels_single[ignore_flag] = -1

            #* 前景点的gt包围框的数据, [前景点个数, 8]
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[box_fg_flag]]
            #* 前景点的类别标签是1
            point_cls_labels_single[box_fg_flag] = 1
            #* 将标签填到对应batch_idx的地方
            point_cls_labels[bs_mask] = point_cls_labels_single
            #* 初始化regression的标签
            point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), 3))
            #* regression的标签的值就是对应的gt框的中心点标签
            point_reg_labels_single[box_fg_flag] = gt_box_of_fg_points[:, 0:3]
            #* 将标签填到对应batch_idx的地方
            point_reg_labels[bs_mask] = point_reg_labels_single

        #* point_cls_labels: [batch_size * vote点的个数]， 用来存储每个vote point的类别
        #*                  1:前景点， 0:背景点， -1:忽略的点
        #* point_reg_labels: [batch_size * vote点的个数, 3],用来存储每个vote point对应的gt框的中心点坐标
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
        }
        return targets_dict

    def assign_targets_simple(self, points, gt_boxes, extra_width=None, set_ignore_flag=True):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extra_width: (dx, dy, dz) extra width applied to gt boxes
            assign_method: binary or distance
            set_ignore_flag:
        Returns:
            point_vote_labels: (N1 + N2 + N3 + ..., 3)
        """
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert points.shape.__len__() in [2], 'points.shape=%s' % str(points.shape)
        batch_size = gt_boxes.shape[0]
        #* extend_gt_boxes: [batch_size, batch中sample的最大包围框个数, 8]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) \
            if extra_width is not None else gt_boxes
        if set_ignore_flag:
            targets_dict = self.assign_stack_targets_simple(points=points, gt_boxes=gt_boxes,
                                                            extend_gt_boxes=extend_gt_boxes,
                                                            set_ignore_flag=set_ignore_flag)
        else:
            targets_dict = self.assign_stack_targets_simple(points=points, gt_boxes=extend_gt_boxes,
                                                            set_ignore_flag=set_ignore_flag)
        return targets_dict
    
    def assign_stack_confident_targets_mask(self, points, pred_boxes, gt_boxes, 
                                            pos_iou_threshold, neg_iou_threshold,
                                            extend_gt_boxes=None,
                                  set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                                  ):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            set_ignore_flag:
            use_ball_constraint:
            central_radius:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((points.shape[0], gt_boxes.size(2) - 1))
        ori_fg_num = 0
        ori_bg_num = 0

        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            pred_boxes_single = pred_boxes[bs_mask]
            pred_boxes_iou = iou3d_nms_utils.boxes_iou3d_gpu(
                pred_boxes_single,
                gt_boxes[k][:, :7]
            )
            pred_boxes_iou, box_idxs_of_pts = torch.max(pred_boxes_iou, dim=-1)
            # fg_flag = pred_boxes_iou > pos_iou_threshold
            # ignore_flag = (pred_boxes_iou > neg_iou_threshold) ^ fg_flag
            #* 计算每个点对应的gt框的索引，没有对应的框就是-1
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            #* 在gt内的flag
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                #* 前景点为在包围框内, 且和gt的iou大于pos_iou_threshold
                
                fg_flag = box_fg_flag & (pred_boxes_iou > pos_iou_threshold)
                ori_fg_num += box_fg_flag.sum()
                #* 背景点为不在包围框内, 且和gt的iou小于neg_iou_threshold, 并且点不在扩大的包围框内
                bg_flag = (~box_fg_flag) & (pred_boxes_iou < neg_iou_threshold) & (extend_box_idxs_of_pts==0)
                ori_bg_num += ((~box_fg_flag) & (extend_box_idxs_of_pts==0)).sum()
                ignore_flag = ~(fg_flag | bg_flag)
                assert (fg_flag.sum() + bg_flag.sum() + ignore_flag.sum() == ignore_flag.shape[0])
                # ignore_flag = (fg_flag ^ (extend_box_idxs_of_pts >= 0)) | ((pred_boxes_iou > neg_iou_threshold) ^ fg_flag)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                #* 前景点对应的gt包围框的中心点坐标
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                #* 该点和这个gt框的中心点坐标需要在半径内
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                #* 既在gt框内，和gt框的中心点距离小于central_radius， fg_flag为1
                fg_flag = box_fg_flag & ball_flag & (pred_boxes_iou > pos_iou_threshold)
                ori_fg_num += (box_fg_flag & ball_flag).sum()
                #* 在gt框内，但是和中心点距离大于central_radius的点设置为忽略点，类别为-1
                #* 背景点为不在包围框内, 且和gt的iou小于neg_iou_threshold
                bg_flag = (~box_fg_flag) & (pred_boxes_iou < neg_iou_threshold)
                ori_bg_num += (~box_fg_flag).sum()
                ignore_flag = ~(fg_flag | bg_flag)
                assert (fg_flag.sum() + bg_flag.sum() + ignore_flag.sum() == ignore_flag.shape[0])
                # ignore_flag = (fg_flag ^ box_fg_flag) | (((pred_boxes_iou > neg_iou_threshold) ^ fg_flag))
                point_cls_labels_single[ignore_flag] = -1
            else:
                raise NotImplementedError
            
            #* 前景点对应的gt包围框
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            #* 前景点生成的包围框的类别
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            #* 对于回归而言，只要在包围框里面的点都可以做回归
            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                #* 根据点的中心点坐标， 前景点对应的gt框的类别和属性得到需要回归的数据
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single
                #* 点对应的gt框
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_boxes.size(2) - 1))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                point_box_labels[bs_mask] = point_box_labels_single

        #* point_cls_labels: 每个点生成的预测框的类别
        #* point_reg_labels: 每个点生成的预测框的回归值
        #* point_box_labels: 每个点生成的预测框对应的gt框的值
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels,
            'ori_fg_num': ori_fg_num,
            'ori_bg_num': ori_bg_num,
        }
        
        return targets_dict


    def assign_stack_targets_mask(self, points, gt_boxes, extend_gt_boxes=None,
                                  set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            set_ignore_flag:
            use_ball_constraint:
            central_radius:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((points.shape[0], gt_boxes.size(2) - 1))
        use_bg_points = self.model_cfg.TARGET_CONFIG.get('USE_BG_POINTS', False)

        if(use_bg_points):
            point_nearest_box_labels = gt_boxes.new_zeros((points.shape[0], gt_boxes.size(2) - 1))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            #* 计算每个点对应的gt框的索引，没有对应的框就是-1
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            #* 前景点的flag
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                #* 前景点对应的gt包围框的中心点坐标
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                #* 该点和这个gt框的中心点坐标需要在半径内
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                #* 既在gt框内，和gt框的中心点距离小于central_radius， fg_flag为1
                fg_flag = box_fg_flag & ball_flag
                #* 在gt框内，但是和中心点距离大于central_radius的点设置为忽略点，类别为-1
                ignore_flag = fg_flag ^ box_fg_flag
                point_cls_labels_single[ignore_flag] = -1
            else:
                raise NotImplementedError
            if(use_bg_points):
                bg_flag = ~fg_flag
                gt_box_centers = gt_boxes[k][:, 0:3].clone()
                distance_mat = torch.cdist(points_single, gt_box_centers, p=2)
                nearest_box_idxs_of_pts = torch.argmin(distance_mat, dim = 1)
                gt_box_of_bg_points = gt_boxes[k][nearest_box_idxs_of_pts[bg_flag]]
                
            #* 前景点对应的gt包围框
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            #* 前景点生成的包围框的类别
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                #* 根据点的中心点坐标， 前景点对应的gt框的类别和属性得到需要回归的数据
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single
                #* 点对应的gt框
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_boxes.size(2) - 1))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                if(use_bg_points):
                    point_nearest_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_boxes.size(2) - 1))
                    point_nearest_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                    point_nearest_box_labels_single[bg_flag] = gt_box_of_bg_points[:, :-1]
                    point_nearest_box_labels[bs_mask] = point_nearest_box_labels_single
                point_box_labels[bs_mask] = point_box_labels_single

        #* point_cls_labels: 每个点生成的预测框的类别
        #* point_reg_labels: 每个点生成的预测框的回归值
        #* point_box_labels: 每个点生成的预测框对应的gt框的值
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels,
            'ori_fg_num': (point_cls_labels>0).sum(),
            'ori_bg_num': (point_cls_labels==0).sum()
        }
        if(use_bg_points):
            targets_dict.update({'point_nearest_box_labels':point_nearest_box_labels})
        return targets_dict

    def assign_stack_targets_iou(self, points, pred_boxes, gt_boxes,
                                 pos_iou_threshold=0.5, neg_iou_threshold=0.35):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            pred_boxes: (N, 7/8)
            gt_boxes: (B, M, 8)
            pos_iou_threshold:
            neg_iou_threshold:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(pred_boxes.shape) == 2 and pred_boxes.shape[1] >= 7, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(pred_boxes.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((pred_boxes.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((pred_boxes.shape[0], 7))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            pred_boxes_single = pred_boxes[bs_mask]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            pred_boxes_iou = iou3d_nms_utils.boxes_iou3d_gpu(
                pred_boxes_single,
                gt_boxes[k][:, :7]
            )
            pred_boxes_iou, box_idxs_of_pts = torch.max(pred_boxes_iou, dim=-1)
            fg_flag = pred_boxes_iou > pos_iou_threshold
            ignore_flag = (pred_boxes_iou > neg_iou_threshold) ^ fg_flag
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels_single[ignore_flag] = -1
            point_cls_labels[bs_mask] = point_cls_labels_single

            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single

                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 7))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                point_box_labels[bs_mask] = point_box_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels
        }
        return targets_dict
    
    def assign_confident_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        assign_method = self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD  # mask or iou
        if assign_method == 'mask':
            points = input_dict['point_vote_coords']
            pred_boxes = input_dict['point_box_preds']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
            pos_iou_threshold = self.model_cfg.TARGET_CONFIG.POS_IOU_THRESHOLD
            neg_iou_threshold = self.model_cfg.TARGET_CONFIG.NEG_IOU_THRESHOLD
            targets_dict = self.assign_stack_confident_targets_mask(
                points=points,  pred_boxes = pred_boxes, gt_boxes=gt_boxes,
                set_ignore_flag=False, use_ball_constraint=True, central_radius=central_radius, 
                pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold,
            )
        elif assign_method == 'iou':
            points = input_dict['point_vote_coords']
            pred_boxes = input_dict['point_box_preds']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert pred_boxes.shape.__len__() == 2, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
            pos_iou_threshold = self.model_cfg.TARGET_CONFIG.POS_IOU_THRESHOLD
            neg_iou_threshold = self.model_cfg.TARGET_CONFIG.NEG_IOU_THRESHOLD
            targets_dict = self.assign_stack_targets_iou(
                points=points, pred_boxes=pred_boxes, gt_boxes=gt_boxes,
                pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold
            )
        else:
            raise NotImplementedError

        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        assign_method = self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD  # mask or iou
        if assign_method == 'mask':
            points = input_dict['point_vote_coords']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
            targets_dict = self.assign_stack_targets_mask(
                points=points, gt_boxes=gt_boxes,
                set_ignore_flag=False, use_ball_constraint=True, central_radius=central_radius, 
            )
        elif assign_method == 'iou':
            points = input_dict['point_vote_coords']
            pred_boxes = input_dict['point_box_preds']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert pred_boxes.shape.__len__() == 2, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
            pos_iou_threshold = self.model_cfg.TARGET_CONFIG.POS_IOU_THRESHOLD
            neg_iou_threshold = self.model_cfg.TARGET_CONFIG.NEG_IOU_THRESHOLD
            targets_dict = self.assign_stack_targets_iou(
                points=points, pred_boxes=pred_boxes, gt_boxes=gt_boxes,
                pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold
            )
        else:
            raise NotImplementedError

        return targets_dict

    def get_vote_layer_loss(self, tb_dict=None):
        #* 大于0说明这个vote point有对应的gt框
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        #* vote_reg_labels: vote point的offset的label [batch_size * vote point个数, 3]
        vote_reg_labels = self.forward_ret_dict['vote_reg_labels']
        #* vote_reg_preds: vote point的offset的预测值 [batch_size * vote point个数, 3]
        vote_reg_preds = self.forward_ret_dict['point_vote_coords']

        #* 最后求出来的loss要除以前景点的个数
        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        vote_loss_reg_src = self.reg_loss_func(
            vote_reg_preds[None, ...],
            vote_reg_labels[None, ...],
            weights=reg_weights[None, ...])
        vote_loss_reg = vote_loss_reg_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        vote_loss_reg = vote_loss_reg * loss_weights_dict['vote_reg_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_reg': vote_loss_reg.item()})
        return vote_loss_reg, tb_dict

    @torch.no_grad()
    def generate_centerness_label(self, point_base, point_box_labels, pos_mask, epsilon=1e-6):
        """
        Args:
            point_base: (N1 + N2 + N3 + ..., 3)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
            pos_mask: (N1 + N2 + N3 + ...)
            epsilon:
        Returns:
            centerness_label: (N1 + N2 + N3 + ...)
        """
        
        use_bg_points = self.model_cfg.TARGET_CONFIG.get('USE_BG_POINTS', False)
        centerness = point_box_labels.new_zeros(pos_mask.shape)
        #* 挑选前景点的包围框标签
        point_box_labels = point_box_labels[pos_mask, :]
        #* 前景点转换到gt包围框坐标系下
        canonical_xyz = point_base[pos_mask, :] - point_box_labels[:, :3]
        rys = point_box_labels[:, -1]
        canonical_xyz = common_utils.rotate_points_along_z(
            canonical_xyz.unsqueeze(dim=1), -rys
        ).squeeze(dim=1)
        #* 计算距离六个面的距离
        distance_front = point_box_labels[:, 3] / 2 - canonical_xyz[:, 0]
        distance_back = point_box_labels[:, 3] / 2 + canonical_xyz[:, 0]
        distance_left = point_box_labels[:, 4] / 2 - canonical_xyz[:, 1]
        distance_right = point_box_labels[:, 4] / 2 + canonical_xyz[:, 1]
        distance_top = point_box_labels[:, 5] / 2 - canonical_xyz[:, 2]
        distance_bottom = point_box_labels[:, 5] / 2 + canonical_xyz[:, 2]

        #* 根据六个面计算centerness
        centerness_l = torch.min(distance_front, distance_back) / torch.max(distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(distance_left, distance_right)
        centerness_h = torch.min(distance_top, distance_bottom) / torch.max(distance_top, distance_bottom)
        #* 这边是1/3次方，不是1/2次方
        if(use_bg_points):
            centerness_pos = torch.sign(centerness_l * centerness_w * centerness_h) * \
                torch.clamp(torch.abs(centerness_l * centerness_w * centerness_h), min=epsilon)**(1/3)
        else:
            centerness_pos = torch.clamp(centerness_l * centerness_w * centerness_h, min=epsilon) ** (1 / 3.0)
        centerness[pos_mask] = centerness_pos

        return centerness

    def get_axis_aligned_iou_loss_lidar(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: (N, 7) float Tensor.
            gt_boxes: (N, 7) float Tensor.
        Returns:
            iou_loss: (N) float Tensor.
        """
        assert pred_boxes.shape[0] == gt_boxes.shape[0]
        #* pos_p: 预测包围框的位置， len_p: 预测包围框的尺寸
        #* pos_g: gt包围框的位置, len_g: gt包围框的尺寸
        pos_p, len_p, *cps = torch.split(pred_boxes, 3, dim=-1)
        pos_g, len_g, *cgs = torch.split(gt_boxes, 3, dim=-1)

        len_p = torch.clamp(len_p, min=1e-5)
        len_g = torch.clamp(len_g, min=1e-5)
        #* vol_p 预测包围框的体积
        vol_p = len_p.prod(dim=-1)
        #* vol_g gt包围框的体积
        vol_g = len_g.prod(dim=-1)
        #* 计算在不考虑航向角的情况下的iou        
        min_p, max_p = pos_p - len_p / 2, pos_p + len_p / 2
        min_g, max_g = pos_g - len_g / 2, pos_g + len_g / 2

        min_max = torch.min(max_p, max_g)
        max_min = torch.max(min_p, min_g)
        diff = torch.clamp(min_max - max_min, min=0)
        intersection = diff.prod(dim=-1)
        union = vol_p + vol_g - intersection
        iou_axis_aligned = intersection / torch.clamp(union, min=1e-5)

        #* loss值是1-axis-aligned-iou
        iou_loss = 1 - iou_axis_aligned
        return iou_loss

    def get_corner_loss_lidar(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: (N, 7) float Tensor.
            gt_boxes: (N, 7) float Tensor.
        Returns:
            corner_loss: (N) float Tensor.
        """
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        pred_box_corners = box_utils.boxes_to_corners_3d(pred_boxes)
        gt_box_corners = box_utils.boxes_to_corners_3d(gt_boxes)

        gt_boxes_flip = gt_boxes.clone()
        gt_boxes_flip[:, 6] += np.pi
        gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_boxes_flip)
        # (N, 8, 3)
        corner_loss = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners, 1.0)
        corner_loss_flip = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners_flip, 1.0)
        corner_loss = torch.min(corner_loss.sum(dim=2), corner_loss_flip.sum(dim=2))

        return corner_loss.mean(dim=1)

    def get_cls_layer_loss(self, tb_dict=None):
        #* 预测的包围框的类别标签: [batch_size * vote point个数]
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        #* 预测的包围框的预测类别: [batch_size * vote point个数, 类别数]
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        #* 标签>0是前景框， ==0是背景框
        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        #* 限制计算的背景点loss个数不超过前景点
        if(self.model_cfg.TARGET_CONFIG.get('RESTRICTED_NEG_NUM', False)):
            neg_num = negatives.sum()
            pos_num = positives.sum()
            if(neg_num > pos_num):
                negatives_idxs = negatives.nonzero().squeeze(-1)
                choosen_idx = np.random.choice(negatives_idxs.shape[0], (neg_num-pos_num).item(), replace=False)
                negatives[negatives_idxs[choosen_idx]] = False
                
        scale_loss = self.model_cfg.TARGET_CONFIG.get('SCALE_LOSS', False)
        ori_fg_num = self.forward_ret_dict['ori_fg_num']
        cur_fg_num = positives.sum()
        ori_bg_num = self.forward_ret_dict['ori_bg_num']
        cur_bg_num = negatives.sum()
        if(scale_loss):
            cls_weights = positives * ori_fg_num/torch.clamp(cur_fg_num, min=1.0) + negatives * ori_bg_num/torch.clamp(cur_bg_num, min=1.0)
        else:
            cls_weights = positives * 1.0 + negatives * 1.0
        #* 转成one-hot编码
        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        self.forward_ret_dict['point_cls_labels_onehot'] = one_hot_targets

        loss_cfgs = self.model_cfg.LOSS_CONFIG
        target_config = self.model_cfg.TARGET_CONFIG
        use_bg_points = target_config.get('USE_BG_POINTS', False)
        if(target_config.CLS_SCORE_TYPE == 'centerness'):
            #* point_base: vote point的坐标 [batch_size * vote point个数, 3]
            point_base = self.forward_ret_dict['point_vote_coords']
            #* point_box_labels: vote point生成的包围框的标签 [batch_size * vote point个数, 7]
            if(use_bg_points):
                point_box_labels = self.forward_ret_dict['point_nearest_box_labels']
                mask = torch.ones(point_base.shape[0], device = point_base.device) >0
                centerness_label = self.generate_centerness_label(point_base, point_box_labels, mask)
                centerness_label = (centerness_label+1)/2
                #* 映射到-1~-0.5 ， 1.5~2
                # centerness_label = (centerness_label) + ((centerness_label>0.5) * 2-1)
            else:
                point_box_labels = self.forward_ret_dict['point_box_labels']
                centerness_label = self.generate_centerness_label(point_base, point_box_labels, positives)
            
            loss_cls_cfg = loss_cfgs.get('LOSS_CLS_CONFIG', None)
            centerness_min = loss_cls_cfg['centerness_min'] if loss_cls_cfg is not None else 0.0
            centerness_max = loss_cls_cfg['centerness_max'] if loss_cls_cfg is not None else 1.0
            centerness_label = centerness_min + (centerness_max - centerness_min) * centerness_label
            #* centerness作为分类预测的标签
        if(use_bg_points):
            one_hot_targets = centerness_label.unsqueeze(dim=-1)
            #! 去除sigmoid
            point_cls_preds = torch.sigmoid(point_cls_preds)
            positive_point_loss_cls = self.cls_loss_func(point_cls_preds[positives], one_hot_targets[positives, :], weights=cls_weights[positives])
            negative_point_loss_cls = self.cls_loss_func(point_cls_preds[negatives], one_hot_targets[negatives, :], weights=cls_weights[negatives])
            point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets[..., :], weights=cls_weights)
            # with open("sasa_exp_version_9.pkl", 'ab') as f:
            #     import pickle
            #     pickle.dump((point_cls_preds, one_hot_targets), f)
            # point_loss_cls_cls = (point_cls_preds>0.5) ^ (one_hot_targets>0.5) + (point_cls_preds<0.5) ^ (one_hot_targets<0.5)
            # point_loss_cls_cls = torch.clamp(- (point_cls_preds- 0.5) * (one_hot_targets-0.5), min = 0)
            # point_loss_cls = point_loss_cls+sum(point_loss_cls_cls)/point_loss_cls_cls.shape[0]
        else:
            one_hot_targets *= centerness_label.unsqueeze(dim=-1)
            positive_point_loss_cls = self.cls_loss_func(point_cls_preds[positives], one_hot_targets[positives, 1:], weights=cls_weights[positives])
            negative_point_loss_cls = self.cls_loss_func(point_cls_preds[negatives], one_hot_targets[negatives, 1:], weights=cls_weights[negatives])
            point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets[..., 1:], weights=cls_weights)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_pos_num': positives.sum().item(),
            'point_negative_num': negatives.sum().item(),
            'positives': positives,
            'negatives': negatives,
            'point_pos_num_differ': (ori_fg_num - cur_fg_num).item(),
            'point_neg_num_differ': (ori_bg_num - cur_bg_num).item(),
        })
        return point_loss_cls, cls_weights, tb_dict
    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_reg_preds = self.forward_ret_dict['point_reg_preds']
        point_reg_labels = self.forward_ret_dict['point_reg_labels']

        ori_pos_num = self.forward_ret_dict['ori_fg_num']
        cur_pos_num = pos_mask.sum()
        scale_loss = self.model_cfg.TARGET_CONFIG.get('SCALE_LOSS', False)
        if(scale_loss):
            reg_weights = pos_mask.float() * ori_pos_num/torch.clamp(cur_pos_num, min=1.0)
        else:
            reg_weights = pos_mask.float()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        if tb_dict is None:
            tb_dict = {}
        #* 前面六个属性(x,y,z,dx,dy,dz)用smoothl1
        point_loss_offset_reg = self.reg_loss_func(
            point_reg_preds[None, :, :6],
            point_reg_labels[None, :, :6],
            weights=reg_weights[None, ...]
        )
        point_loss_offset_reg = point_loss_offset_reg.sum(dim=-1).squeeze()

        if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
            point_loss_velo_reg = self.reg_loss_func(
                point_reg_preds[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                point_reg_labels[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                weights=reg_weights[None, ...]
            )
            point_loss_velo_reg = point_loss_velo_reg.sum(dim=-1).squeeze()
            point_loss_offset_reg = point_loss_offset_reg + point_loss_velo_reg

        point_loss_offset_reg *= loss_weights_dict['point_offset_reg_weight']

        if isinstance(self.box_coder, box_coder_utils.PointBinResidualCoder):
            #* 预测的角度bin的标签
            point_angle_cls_labels = \
                point_reg_labels[:, 6:6 + self.box_coder.angle_bin_num]
            #* 使用cross entropy计算
            #! 其中第二个参数target送进去的是每个量的label的索引，并不是one-hot
            point_loss_angle_cls = F.cross_entropy(  # angle bin cls
                point_reg_preds[:, 6:6 + self.box_coder.angle_bin_num],
                point_angle_cls_labels.argmax(dim=-1), reduction='none') * reg_weights
            #* 将12个regression都取出来
            point_angle_reg_preds = point_reg_preds[:, 6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
            point_angle_reg_labels = point_reg_labels[:, 6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
            #* 只取标签那一类的regression
            point_angle_reg_preds = (point_angle_reg_preds * point_angle_cls_labels).sum(dim=-1, keepdim=True)
            point_angle_reg_labels = (point_angle_reg_labels * point_angle_cls_labels).sum(dim=-1, keepdim=True)
            point_loss_angle_reg = self.reg_loss_func(
                point_angle_reg_preds[None, ...],
                point_angle_reg_labels[None, ...],
                weights=reg_weights[None, ...]
            )
            point_loss_angle_reg = point_loss_angle_reg.squeeze()

            point_loss_angle_cls *= loss_weights_dict['point_angle_cls_weight']
            point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']

            point_loss_box = point_loss_offset_reg + point_loss_angle_cls + point_loss_angle_reg  # (N)
        else:
            point_angle_reg_preds = point_reg_preds[:, 6:]
            point_angle_reg_labels = point_reg_labels[:, 6:]
            point_loss_angle_reg = self.reg_loss_func(
                point_angle_reg_preds[None, ...],
                point_angle_reg_labels[None, ...],
                weights=reg_weights[None, ...]
            )
            point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']
            point_loss_box = point_loss_offset_reg + point_loss_angle_reg
        #* 如果前景框个数>0
        if reg_weights.sum() > 0:
            point_box_preds = self.forward_ret_dict['point_box_preds']
            point_box_labels = self.forward_ret_dict['point_box_labels']
            point_loss_box_aux = 0

            if self.model_cfg.LOSS_CONFIG.get('AXIS_ALIGNED_IOU_LOSS_REGULARIZATION', False):
                point_loss_iou = self.get_axis_aligned_iou_loss_lidar(
                    point_box_preds[pos_mask, :],
                    point_box_labels[pos_mask, :]
                )
                point_loss_iou *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_iou_weight']
                point_loss_box_aux = point_loss_box_aux + point_loss_iou

            if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
                #* corner loss
                point_loss_corner = self.get_corner_loss_lidar(
                    point_box_preds[pos_mask, 0:7],
                    point_box_labels[pos_mask, 0:7]
                )
                point_loss_corner *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_corner_weight']
                point_loss_box_aux = point_loss_box_aux + point_loss_corner
            
            point_loss_box[pos_mask] = point_loss_box[pos_mask] + point_loss_box_aux

        return point_loss_box, reg_weights, tb_dict  # point_loss_box: (N)

    def get_sasa_layer_loss(self, tb_dict=None):
        if self.enable_sasa:
            point_loss_sasa_list = self.loss_point_sasa.loss_forward(
                self.forward_ret_dict['point_sasa_preds'],
                self.forward_ret_dict['point_sasa_labels']
            )
            point_loss_sasa = 0
            tb_dict = dict()
            for i in range(len(point_loss_sasa_list)):
                cur_point_loss_sasa = point_loss_sasa_list[i]
                if cur_point_loss_sasa is None:
                    continue
                point_loss_sasa = point_loss_sasa + cur_point_loss_sasa
                tb_dict['point_loss_sasa_layer_%d' % i] = point_loss_sasa_list[i].item()
            tb_dict['point_loss_sasa'] = point_loss_sasa.item()
            return point_loss_sasa, tb_dict
        else:
            return None, None

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_vote, tb_dict_0 = self.get_vote_layer_loss()
        
        point_loss_cls, cls_weights, tb_dict_1 = self.get_cls_layer_loss()
        
        differ_pos_and_neg = self.model_cfg.TARGET_CONFIG.get('DIFFER_POS_AND_NEG', False)
        point_loss_box, box_weights, tb_dict_2 = self.get_box_layer_loss()
        positives = tb_dict_1.pop('positives')
        negatives = tb_dict_1.pop('negatives')
        if(differ_pos_and_neg):
            pos_point_loss_cls = point_loss_cls[positives].sum() / torch.clamp(positives.sum(), min=1.0)
            negative_point_loss_cls = point_loss_cls[negatives].sum() / torch.clamp(negatives.sum(), min=1.0)
            point_loss_box = point_loss_box.sum() / torch.clamp(box_weights.sum(), min=1.0)
            
            tb_dict.update({
                'point_loss_vote': point_loss_vote.item(),
                'pos_point_loss_cls': pos_point_loss_cls.item(),
                'neg_point_loss_cls': negative_point_loss_cls.item(),
                'point_loss_box': point_loss_box.item(),

            })
        else:
            point_loss_cls = point_loss_cls.sum() / torch.clamp(cls_weights.sum(), min=1.0)
            point_loss_box = point_loss_box.sum() / torch.clamp(box_weights.sum(), min=1.0)
            tb_dict.update({
                'point_loss_vote': point_loss_vote.item(),
                'point_loss_cls': point_loss_cls.item(),
                'point_loss_box': point_loss_box.item(),
            })

        point_loss = point_loss_vote + point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_0)
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)

        point_loss_sasa, tb_dict_3 = self.get_sasa_layer_loss()
        if point_loss_sasa is not None:
            tb_dict.update(tb_dict_3)
            point_loss += point_loss_sasa
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_scores (optional): (B, N)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        batch_size = batch_dict['batch_size']
        #* 最后一层SA模块输出的采样点坐标, [batch_size * 最后一个SA模块的采样点个数, 4] 4:batch_idx, x, y, z
        point_coords = batch_dict['point_coords']
        #* 最后一层SA模块输出的采样点特征, [batch_size * 最后一个SA模块的采样点个数, 特征维度]
        point_features = batch_dict['point_features']

        batch_idx, point_coords = point_coords[:, 0], point_coords[:, 1:4]
        #* batch_idx: [batch_size * 最后一个SA模块的采样点个数] -> [batch_size, 最后一个SA模块的采样点个数, 1]
        batch_idx = batch_idx.view(batch_size, -1, 1)
        #* point_coords: [batch_size * 最后一个SA模块的采样点个数, 3] -> [batch_size, 最后一个SA模块的采样点个数, 3]
        point_coords = point_coords.view(batch_size, -1, 3).contiguous()
        #* point_features: [batch_size * 最后一个SA模块的采样点个数, 特征维度] 
        #* -> [batch_size, 最后一个SA模块的采样点个数, 特征维度]
        #* -> [batch_size, 特征维度, 最后一个SA模块的采样点个数]
        point_features = point_features.reshape(
            batch_size,
            point_coords.size(1),
            -1
        ).permute(0, 2, 1).contiguous()

        # candidate points sampling
        #* 一般是基于特征采样的采样范围
        sample_range = self.model_cfg.SAMPLE_RANGE
        #* 候选点的batch_idx
        sample_batch_idx = batch_idx[:, sample_range[0]:sample_range[1], :].contiguous()
        #* 候选点的坐标
        candidate_coords = point_coords[:, sample_range[0]:sample_range[1], :].contiguous()
        #* 候选点的特征
        candidate_features = point_features[:, :, sample_range[0]:sample_range[1]].contiguous()

        # generate vote points
        #* vote_offsets:[batch_size, 3, candidate points的个数]
        vote_offsets = self.vote_layers(candidate_features)  # (B, 3, N)
        #* vote_offsets必须在一定范围内
        vote_translation_range = np.array(self.vote_cfg.MAX_TRANSLATION_RANGE, dtype=np.float32)
        vote_translation_range = torch.from_numpy(vote_translation_range).cuda().unsqueeze(dim=0).unsqueeze(dim=-1)
        vote_offsets = torch.max(vote_offsets, -vote_translation_range)
        vote_offsets = torch.min(vote_offsets, vote_translation_range)
        vote_coords = candidate_coords + vote_offsets.permute(0, 2, 1).contiguous()

        ret_dict = {'batch_size': batch_size,
                    'point_candidate_coords': candidate_coords.view(-1, 3).contiguous(),
                    'point_vote_coords': vote_coords.view(-1, 3).contiguous()}
        #* [batch_size， candidate points的个数, 1] -> [batch_size*candidate points的个数, 1]
        sample_batch_idx_flatten = sample_batch_idx.view(-1, 1).contiguous()  # (N, 1)
        #*[batch_size*candidate points的个数, 1] -> [batch_size*candidate points的个数]
        batch_dict['batch_index'] = sample_batch_idx_flatten.squeeze(-1)
        #* batch_dict['point_candidate_coords']： [batch_size*candidate points的个数, 4]
        batch_dict['point_candidate_coords'] = torch.cat(  # (N, 4)
            (sample_batch_idx_flatten, ret_dict['point_candidate_coords']), dim=-1)
        #*batch_dict['point_vote_coords']： [batch_size*candidate points的个数, 4]
        batch_dict['point_vote_coords'] = torch.cat(  # (N, 4)
            (sample_batch_idx_flatten, ret_dict['point_vote_coords']), dim=-1)

        if self.training:  # assign targets for vote loss
            #* vote point和扩大后的包围框做匹配
            extra_width = self.model_cfg.TARGET_CONFIG.get('VOTE_EXTRA_WIDTH', None)
            targets_dict = self.assign_targets_simple(batch_dict['point_candidate_coords'],
                                                      batch_dict['gt_boxes'],
                                                      extra_width=extra_width,
                                                      set_ignore_flag=False)
            #* ret_dict['vote_cls_labels']: [batch_size * vote点的个数]， 用来存储每个vote point的类别
            #*                              1:前景点， 0:背景点， -1:忽略的点
            #* ret_dict['vote_reg_labels']: [batch_size * vote点的个数, 3],用来存储每个vote point对应的gt框的中心点坐标
            ret_dict['vote_cls_labels'] = targets_dict['point_cls_labels']  # (N)
            ret_dict['vote_reg_labels'] = targets_dict['point_reg_labels']  # (N, 3)

        
        #* 对vote point调用SA模块, 这个直接把new_xyz传进去，就不会做采样了
        #* point_coords是最后一个SA模块采样的点的坐标
        #* 输入的point_features是最后一个SA模块采样的点的特征
        #* 输出的point_feature，[batch_size, 点的特征, vote point的个数]
        _, point_features, _, idx_cnt = self.SA_module(
            point_coords,
            point_features,
            new_xyz=vote_coords
        )
        #* 经过一连串的conv1d -> bn -> relu
        #* point_features[batch_size, 特征维度， vote_point的个数]
        point_features = self.shared_fc_layer(point_features)
        #* 经过一连串的conv1d -> bn -> relu 和一个 conv1d
        #* point_cls_preds[batch_size, 类别个数， vote_point的个数]
        point_cls_preds = self.cls_layers(point_features)
        #* 经过一连串的conv1d -> bn -> relu 和一个 conv1d
        #* point_reg_preds[batch_size, 回归数， vote_point的个数]， 这边是对航向角划分了12个bin，最后是6+12*2=30
        point_reg_preds = self.reg_layers(point_features)
        
        #* point_cls_preds:
        #* [batch_size, 类别个数， vote_point的个数]
        #* -> [batch_size, vote_point的个数, 类别个数]
        #* -> [batch_size*vote_point的个数, 类别个数]
        point_cls_preds = point_cls_preds.permute(0, 2, 1).contiguous()
        point_cls_preds = point_cls_preds.view(-1, point_cls_preds.shape[-1]).contiguous()
        
        #* point_reg_preds:
        #* [batch_size, 回归数， vote_point的个数]
        #* -> [batch_size, vote_point的个数, 回归数]
        #* -> [batch_size*vote_point的个数, 回归数]
        point_reg_preds = point_reg_preds.permute(0, 2, 1).contiguous()
        point_reg_preds = point_reg_preds.view(-1, point_reg_preds.shape[-1]).contiguous()

        #* 对类别的分数求sigmoid
        point_cls_scores = torch.sigmoid(point_cls_preds)
        #* batch_dict['point_cls_scores']： vote point产生的包围框预测的分数 [batch_size * vote point的个数, 类别数]
        batch_dict['point_cls_scores'] = point_cls_scores
        #* 根据预测值还原包围框数据
        point_box_preds = self.box_coder.decode_torch(point_reg_preds,
                                                      ret_dict['point_vote_coords'])
        #* batch_dict['point_box_preds']:还原的预测包围框数据 [batch_size * vote point的个数, 7]
        batch_dict['point_box_preds'] = point_box_preds

        ret_dict.update({'point_cls_preds': point_cls_preds,
                         'point_reg_preds': point_reg_preds,
                         'point_box_preds': point_box_preds,
                         'point_cls_scores': point_cls_scores})
        
        use_bg_points = self.model_cfg.TARGET_CONFIG.get('USE_BG_POINTS', False)
        if self.training:
            if(use_bg_points):
                targets_dict = self.assign_targets(batch_dict)
                if('point_nearest_box_labels' in targets_dict):
                    ret_dict['point_nearest_box_labels'] = targets_dict['point_nearest_box_labels']
            else:
                #* 给预测的包围框赋予对应的gt框
                targets_dict = self.assign_targets(batch_dict)
            #* 每个点生成的预测框的类别，0为背景框， -1为忽略框， 其他为对应的类别
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            #* 每个点生成的预测框的回归真值
            ret_dict['point_reg_labels'] = targets_dict['point_reg_labels']
            #* 每个点生成的预测框对应的gt框的属性
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['ori_fg_num'] = targets_dict['ori_fg_num']
            ret_dict['ori_bg_num'] = targets_dict['ori_bg_num']

            if self.enable_sasa:
                point_sasa_labels = self.loss_point_sasa(
                    batch_dict['point_coords_list'],
                    batch_dict['point_scores_list'],
                    batch_dict['gt_boxes']
                )
                ret_dict.update({
                    'point_sasa_preds': batch_dict['point_scores_list'],
                    'point_sasa_labels': point_sasa_labels
                })

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_vote_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_reg_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['cls_preds_normalized'] = False
        
        if(self.only_cls_confident_points):
            targets_dict = self.assign_confident_targets(batch_dict)
            if(self.confident_points_flag or (targets_dict['point_cls_labels']>0).sum() > self.confident_points_per_batch * batch_dict['batch_size']):
                self.confident_points_flag = True
                #* 每个点生成的预测框的类别，0为背景框， -1为忽略框， 其他为对应的类别
                ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
                #* 每个点生成的预测框的回归真值
                ret_dict['point_reg_labels'] = targets_dict['point_reg_labels']
                #* 每个点生成的预测框对应的gt框的属性
                ret_dict['point_box_labels'] = targets_dict['point_box_labels']
                ret_dict['ori_fg_num'] = targets_dict['ori_fg_num']
                ret_dict['ori_bg_num'] = targets_dict['ori_bg_num']
        self.forward_ret_dict = ret_dict

        return batch_dict
