import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """初始化PFNLayer

        Args:
            in_channels (_type_): 输入通道数
            out_channels (_type_): 输出通道数
            use_norm (bool, optional): 使用使用batchnorm, 如果为True, 每个linear层后面需要跟一个BatchNorm1d. Defaults to True.
            last_layer (bool, optional): 是否是最后一层, 如果不是最后一层, 输出通道数要减半, 因为要把maxpooling结合和它做拼接. Defaults to False.
        """
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        """
        if use_norm and not last layer:
            linear + BN + relu = feature1
            maxpooling(feature1) = feature2
            return concat(feature1, feature2)
        elif use norm and last layer :
            return linear + BN + relu + maxpooling
        elif not use norm and last layer:
            return linear + relu + maxpooling
        elif not use norm and not last layer:
            linear + relu = feature1
            maxpooling(feature1) = feature2
            return concat(feature1, feature2)
        """
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        """初始化PillarVFE

        Args:
            model_cfg (_type_): 模型配置
                USE_NORM: 是否在PFNLayer中, linear后面使用BatchNorm1d
                WITH_DISTANCE: 是否将点云距离原点距离加入模型的输入中
                USE_ABSLOTE_XYZ: 是否将点云的坐标加入模型的输入中
                NUM_FILTERS: 卷积核的中心层特征维度
            num_point_features (_type_): 点云的特征维度
            voxel_size (_type_): voxelsize, [dx, dy, dz]
            point_cloud_range (_type_): 点云范围, [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """模型的forward函数

        Args:
            batch_dict (_type_): 输入的字典
                frame_id:帧id
                gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
                points:点云
                flip_x:是否绕着X轴进行翻转
                noise_rot: 整片点云逆时针旋转的角度
                noise_scale: 整片点云缩放的尺度
                use_lead_xyz: 是否使用xyz数据
                voxels: 非空voxel个数(不超过最大voxel个数) * voxel中最大点个数 * 点特征维度
                voxel_coods: voxel的索引, 非空voxel个数 * 4, [zidx, yidx, xidx, batch_idx]
                voxel_num_points: 非空voxel中的点个数(不超过voxel中最大点个数), 非空voxel个数
                image_shape: 图像尺寸
                batch_size: batch_size

        Returns:
            _type_: _description_
        """
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        #* 求每个voxel中点的坐标的均值
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        #* f_cluster是每个voxel中点的坐标相对于voxel均值点的相对坐标
        f_cluster = voxel_features[:, :, :3] - points_mean

        #* f_center是每个voxel中点的坐标相对于voxel中心点的相对坐标
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        #* voxel_count是voxel中的最大点个数
        voxel_count = features.shape[1]
        #* mask的shape为非空voxel个数 * 每个voxel中的最大点数 * 1
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        #* 这样的话, 补0的点的特征会全部变为0, 因为之前减去了中心点坐标等，所以补0的特征在乘掩码前不是0
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        """            
        返回的batch_dict
                frame_id:帧id
                gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
                points:点云
                flip_x:是否绕着X轴进行翻转
                noise_rot: 整片点云逆时针旋转的角度
                noise_scale: 整片点云缩放的尺度
                use_lead_xyz: 是否使用xyz数据
                voxels: 非空voxel个数(不超过最大voxel个数) * voxel中最大点个数 * 点特征维度
                voxel_coods: voxel的索引, 非空voxel个数 * 4, [zidx, yidx, xidx, batch_idx]
                voxel_num_points: 非空voxel中的点个数(不超过voxel中最大点个数), 非空voxel个数
                image_shape: 图像尺寸
                batch_size: batch_size
                pillar_features: 非空voxel个数 * 特征维度的voxel特征
        """
        return batch_dict
