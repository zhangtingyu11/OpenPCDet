import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        """初始化BaseBEVBackbone

        Args:
            model_cfg (_type_): 模型配置
                LAYER_NUMS: 每个block中的中心层的个数, 中间层的前面会有一层进行特征维度的变换(记为起始层, 在起始层做特征变换和尺寸变化)
                LAYER_STRIDES: 起始层的步长
                NUM_FILTERS: 每个block的输出特征维度
                UPSAMPLE_STRIDES: 上采样的步长
                NUM_UPSAMPLE_FILTERS: 上采样的输出通道数
            input_channels (_type_): 输入通道数
        """
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        #* 输入通道数的列表
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            """self.blocks的网络如下:
            对于第i个block:
                起始层:做特征变化, stride为LAYER_STRIDES[i], 输出通道数NUM_FILTERS[i] 
                        Conv2d -> BN2d -> ReLU
                中间层: (Conv2d -> BN2d -> ReLU) * LAYER_NUMS[i]
            """
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                """self.deblocks的网络如下:
                    对于第i个deblock:
                        做特征变化, stride为UPSAMPLE_STRIDES[i], 输入通道数为NUM_FILTERS[i], 输出通道数为NUM_UPSAMPLE_FILTERS[i]
                                  kernel_size和stride一致
                                deConv2d(如果stride是1, 就是Conv2d) -> BN2d -> ReLU
                """
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
            blocks进行尺寸缩放, 每次缩放完后特征都要放大两倍, 然后通过deblock还原成原图尺寸
            最后将所有deblock的输出进行拼接
        Args:
            data_dict:
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
                spatial_features: BEV特征图, batch_size * 特征维度 * H * W
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        """
        Returns:
        data_dict:
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
            spatial_features: BEV特征图, batch_size * 特征维度 * H * W
            spatial_features_2d: 骨干网络输出的二维特征图
        """
        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
