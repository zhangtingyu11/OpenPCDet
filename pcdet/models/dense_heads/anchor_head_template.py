import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        """初始化AnchorHeadTemplate

        Args:
            model_cfg (_type_): 模型配置
                CLASS_AGNOSTIC:
                USE_DIRECTION_CLASSIFIER:
                DIR_OFFSET:
                DIR_LIMIT_OFFSET:
                NUM_DIR_BINS:
                ANCHOR_GENERATOR_CONFIG: 生成anchor的配置
                    class_name: 类别名,
                    anchor_sizes: anchor大小的取值范围,
                    anchor_rotations: anchor航向角的取值范围,
                    anchor_bottom_heights: anchor的底部高度,
                    align_center: 如果为True, anchor会生成在grid中心, 会以边界为开头和结尾均匀生成,
                    feature_map_stride: 特征图下采样的倍率,
                    matched_threshold' TODO,
                    unmatched_threshold: TODO
                TARGET_ASSIGNER_CONFIG:
                    NAME:
                    POS_FRACTION:
                    SAMPLE_SIZE:
                    NORM_BY_NUM_EXAMPLES:
                    MATCH_HEIGHT:
                    BOX_CODER:
                LOSS_CONFIG:
                    LOSS_WEIGHTS:
            num_class (_type_): 需要预测的类别数量
            class_names (_type_): 需要预测的类别名字
            grid_size (_type_): xyz轴的voxel个数
            point_cloud_range (_type_): 点云范围, [xmin, ymin, zmin, xmax, ymax, zmax]
            predict_boxes_when_training (_type_): 是否在训练时预测包围框
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        """生成anchor

        Args:
            anchor_generator_cfg (_type_): 生成anchor的配置, 是一个列表, 列表中的每个元素是个字典，如下
                class_name: 类别名,
                anchor_sizes: anchor大小的取值范围,
                anchor_rotations: anchor航向角的取值范围,
                anchor_bottom_heights: anchor的底部高度,
                align_center: 如果为True, anchor会生成在grid中心, 会以边界为开头和结尾均匀生成,
                feature_map_stride: 特征图下采样的倍率,
                matched_threshold' TODO,
                unmatched_threshold: TODO

            grid_size (_type_): xyz方向voxel的个数
            point_cloud_range (_type_): 点云范围, [xmin, ymin, zmin, xmax, ymax, zmax]
            anchor_ndim (int, optional): 用多少个维度来描述包围框. Defaults to 7.

        Returns:
            anchors_list: anchor列表, 列表中每一类元素表示那一类的anchor
            num_anchors_per_location_list: 每个位置生成的anchor个数, list, 列表中每个元素表示这个类别生成的anchor个数
        """
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            #* 如果anchor_ndim不为7, 需要将anchor的维度补到anchor_ndim
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        """                
            box_cls_labels: [batch_size, 所有类别的anchor个数和], 0表示背景, >0表示前景, -1表示忽略
            box_reg_targets: [batch_size, 所有类别的anchor个数和, 7], 只对前景求regression, 其他的都是0
            reg_weights: [batch_size, 所有类别的anchor个数和], 前景点的weight是1
        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        #* cls_preds是[batch_size, H, W, num_class*num_class*anchor_num], 每个位置anchor_num个anchor
        cls_preds = self.forward_ret_dict['cls_preds']
        #* box_cls_labels是[batch_size, H*W*num_class*anchor_num]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        #* -1表示忽略, >1表示前景样本, 0表示背景样本
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        #* 前景和背景样本的权重都是1, 忽略的样本的权重为0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        #* pos_normalizer表示前景样本的个数
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        #* 每个样本的权重 * 1/前景样本个数
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        #* 将忽略的样本的label也变成了0
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        #* one_hot_targets为[batch_size, H*W*num_class*anchor_num, 4]
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        #* 将类别填充到one-hot编码中
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        #* [batch_size, H, W, num_class*num_class*anchor_num]-> [batch_size, H*W*num_class*anchor_num, num_class]
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        #* 不取背景类那一维
        one_hot_targets = one_hot_targets[..., 1:]
        #* 分类的权重是前景和背景框的权重为1, 其他的权重为0
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        #* cls_loss是每个样本的loss, 不是一个batch的loss
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        #* anchor的航向角 + target angle offset可以得到gt航向角
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        #* gt航向角-pi/4然后转换到0~2pi
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        #* box_preds为[batch_size, H, W, num_class*box_code_size*anchor_num]
        box_preds = self.forward_ret_dict['box_preds']
        #* box_dir_cls_preds为[batch_size, H, W, num_class*angle_bin*anchor_num]
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        #* box_dir_cls_preds为[batch_size, H*W*num_class*anchor_num, box_code_size]
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        #* box_cls_labels为[batch_size, H*W*num_class*anchor_num]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        #* anchors: [1, H, W, num_class, anchor_num, box_code_size]->[batch_size, H*W*num_class*anchor_num, box_code_size]
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        #* box_preds: [batch_size, H, W, num_class*box_code_size*anchor_num]->[batch_size, H*W*num_class*anchor_num, box_code_size]
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        #* 将pred的航向角变为sinacosb, gt的航向角为cosasinb, 这样两个相减为0，当两个相反的时候也会相等
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            #* 构建onehot编码的标签
            #* 要加DIR_OFFSET的原因是, 如果angle bin是2的话，那么划分的结果为0~pi和pi~2pi, 以我们的经验，一般车是0或者pi, 行人是pi/2或者3pi/2
            #* 为了使得两个样本比较均匀，选择pi/4, 这样预测的值不会出现在边界上
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

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
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
