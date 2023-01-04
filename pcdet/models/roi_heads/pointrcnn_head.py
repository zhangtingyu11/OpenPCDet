import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PointRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        """初始化PointRCNNHead

        Args:
            input_channels (_type_): 输入通道数
            model_cfg (_type_): 
                CLASS_AGNOSTIC: 
                ROI_POINT_POOL:
                    POOL_EXTRA_WIDTH: 
                    NUM_SAMPLED_POINTS: 
                    DEPTH_NORMALIZER: 
                XYZ_UP_LAYER: 
                CLS_FC: 
                REG_FC: 
                DP_RATIO: 
                USE_BN: 
                SA_CONFIG:
                    NPOINTS: 
                    RADIUS: 
                    NSAMPLE: 
                    MLPS: 
                NMS_CONFIG:
                    TRAIN:
                        NMS_TYPE: NMS的类型
                        MULTI_CLASSES_NMS: 
                        NMS_PRE_MAXSIZE: 先挑选多少个置信度最高的预测框
                        NMS_POST_MAXSIZE: 最终保留多少个预测框
                        NMS_THRESH: NMS里IOU的阈值
                    TEST:
                        NMS_TYPE: 
                        MULTI_CLASSES_NMS: 
                        NMS_PRE_MAXSIZE: 
                        NMS_POST_MAXSIZE: 
                        NMS_THRESH: 
                TARGET_CONFIG:
                    BOX_CODER: 
                    ROI_PER_IMAGE: 每个样本需要下采样多少个roi进行训练(不是拿全部的roi进行训练)
                    FG_RATIO: 前景框的比例
                    SAMPLE_ROI_BY_EACH_CLASS: 如果是True则按照类别进行sample
                    CLS_SCORE_TYPE: 分类分数的类型
                    CLS_FG_THRESH: 前景框的分类阈值
                    CLS_BG_THRESH: 背景框的分类阈值
                    CLS_BG_THRESH_LO: 简单背景框的分类阈值
                    HARD_BG_RATIO: 困难背景框的比例
                    REG_FG_THRESH: 
                LOSS_CONFIG:
                    CLS_LOSS: 
                    REG_LOSS: 
                    CORNER_LOSS_REGULARIZATION: 
                    LOSS_WEIGHTS: {
                        'rcnn_cls_weight': 
                        'rcnn_reg_weight': 
                        'rcnn_corner_weight': 
                        'code_weights':
                    }
            num_class (int, optional): _description_. Defaults to 1.
        """
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
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
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict):
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
                rois: (batch_size, 采样后的roi个数, 7)的roi包围框
                roi_scores: (batch_size, 未采样的roi个数)的roi分数
                roi_labels: (batch_size, 采样后的roi个数)的类别标签
                has_class_labels: True表示roi_labels里面按照类别分不同的标签

        """
        batch_size = batch_dict['batch_size']
        #* batch_idx为[B*N], 表示每个点的batch_idx
        batch_idx = batch_dict['point_coords'][:, 0]
        #* point_coords为[B*N, 3], 表示每个点的三维坐标
        point_coords = batch_dict['point_coords'][:, 1:4]
        #* point_features为[B*N, C]表示每个点的特征
        point_features = batch_dict['point_features']
        #* rois为[batch_size, roi个数, 7], 表示roi包围框
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        #* point_scores为[B*N], 表示每个点的分类分数
        point_scores = batch_dict['point_cls_scores'].detach()
        #* 点的深度为距离/DEPTH_NORMALIZER-0.5, 应该是为了让point_depths的范围在-0.5~0.5之间
        #* point_depths为[B*N]
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        #* point_features_all为[B*N, C+2]
        point_features_all = torch.cat(point_features_list, dim=1)
        #* batch_points为[B, N, 3], 表示每个点的坐标
        batch_points = point_coords.view(batch_size, -1, 3)
        #* batch_point_features为[B, N, C], 表示每个点的特征
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            #* pooled_features: [B, roi个数, roi中采样的点数, C]的roi特征, C的前3维是点云的坐标
            #* pooled_empty_flag: [B, roi个数], 如果为1说明这个roi中没有点云, 否则有点云
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            #* 转化到roi坐标系下
            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            #* roi中没有点的roi的特征为0
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        """
        Returns:
            pooled_features: [B*roi个数, roi中采样的点数, C], 其中C的前三维是点的坐标
        """
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C)

        #* [B*roi个数， 5， roi中采样的点数, 1], 其中5为xyz+points_score+points_depth
        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        #* 将点的初始信息转化到和点的特征维度一样
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        #* 拼接, 对特征维度进行降维
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]  # (total_rois, num_features, 1)
        #* rcnn_cls: [B*roi个数, 1]
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)4
        #* rcnn_reg: [B*roi个数, 7]
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            
            """
            self.forward_ret_dict:
                rois: roi包围框, (batch_size, ROI_PER_IMAGE, box_code_size)
                gt_of_rois: roi对应的gt框, 将航向角转换到-pi/2~pi/2, (batch_size, ROI_PER_IMAGE, box_code_size+1), 最后一维是gt框的类别,
                            同时坐标系转化到了anchor坐标系下
                gt_iou_of_rois: roi和其对应的gt的iou, (batch_size, ROI_PER_IMAGE)
                roi_scores: roi的置信度, (batch_size, ROI_PER_IMAGE)
                roi_labels: roi对应的分类标签, (batch_size, ROI_PER_IMAGE)
                reg_valid_mask: roi是否需要计算regression loss的掩码, (batch_size, ROI_PER_IMAGE)
                rcnn_cls_labels: roi的前背景分类, 1为前景, 0为背景, -1为忽略, [B, roi个数]
                gt_of_rois_src: roi对应的gt框, 航向角是原来的0~pi*2, (batch_size, ROI_PER_IMAGE, box_code_size+1), 最后一维是gt框的类别
                rcnn_cls: [B*roi个数, 1], 只判断是前景框还是背景框
                rcnn_reg: [B*roi个数, 7], 包围框的回归参数
            """
            self.forward_ret_dict = targets_dict
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
                cls_preds_normalized: 
                rois: (batch_size, 采样后的roi个数, 7)的roi包围框
                roi_scores: (batch_size, 未采样的roi个数)的roi分数
                roi_labels: (batch_size, 采样后的roi个数)的类别标签
                has_class_labels: True表示roi_labels里面按照类别分不同的标签
        """
        return batch_dict
