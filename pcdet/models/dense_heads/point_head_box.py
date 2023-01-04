import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate


class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        """初始化PointHeadBox

        Args:
            num_class (_type_): 类别数
            input_channels (_type_): 输入通道数
            model_cfg (_type_): 模型配置
                CLS_FC: 分类的全连接层的隐藏层
                REG_FC: 回归的全连接层的隐藏层
                CLASS_AGNOSTIC:
                USE_POINT_FEATURES_BEFORE_FUSION:
                TARGET_CONFIG:
                    GT_EXTRA_WIDTH:
                    BOX_CODER: PointResidualCoder
                    BOX_CODER_CONFIG: 
                        use_mean_size:
                        mean_size:
                LOSS_CONFIG:
                    LOSS_REG:
                    LOSS_WEIGHTS:
                        point_cls_weight:
                        point_box_weight:
                        code_weights:
            predict_boxes_when_training (bool, optional): _description_. Defaults to False.
        """
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        #* point_coords为[batch_size*N, 4], 4个维度为[batch_idx, x, y, z]
        point_coords = input_dict['point_coords']
        #* gt_boxes为[batch_size, gt_boxes个数, 8], 8维为[x, y, z, dx, dy, dz, heading, class_idx]
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        #* 扩大gt包围框的长宽高
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        """
        Returns:
            tb_dict: 
                point_loss_cls: 点的分类loss
                point_pos_num: 前景点的个数
                point_loss_box: 点的回归loss
        """
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            #* point_features为[batch_size*N, C]
            point_features = batch_dict['point_features']
        #* point_cls_preds为[batch_size*N, 3]
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        #* point_box_preds为[batch_size*N, box_code_size]
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
        
        #* 挑选最大的分类分数
        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        #* 将分类分数作用sigmoid
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False
        """
            self.forward_ret_dict: 
                point_cls_preds: 点的预测值, [batch_size*点数, 3]
                point_box_preds: 包围框的预测值, [batch_size*点数, 8], [x, y, z, dx, dy, dz, sina, cosa]
                point_cls_labels: 点的分类label, [batch_size*点数], 0是背景, >0是前景点, -1是忽略的点
                point_box_labels: 包围框的label, [batch_size*点数, 8], [x, y, z, dx, dy, dz, cosa, sina]
        """
        self.forward_ret_dict = ret_dict
        """
        Returns:
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
                batch_index: (B*N), 点的batch_idx
                cls_preds_normalized: 
        """
        return batch_dict
