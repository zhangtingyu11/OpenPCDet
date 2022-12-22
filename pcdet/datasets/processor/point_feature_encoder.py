import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz
       
        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            max_sweeps = self.point_encoding_config.max_sweeps
            idx = self.src_feature_list.index('timestamp')
            dt = np.round(data_dict['points'][:, idx], 2)
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt))-1, max_sweeps-1)]
            data_dict['points'] = data_dict['points'][dt <= max_dt]
        """
        返回的data_dict
            frame_id:帧id
            gt_names:gt框的类别名
            gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
            points:点云
            flip_x:是否绕着X轴进行翻转
            noise_rot: 整片点云逆时针旋转的角度
            noise_scale: 整片点云缩放的尺度
            use_lead_xyz: 是否使用xyz数据
        """
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        """将点云进行编码, 将self.used_feature_list的特征拼接起来, 作为模型的输入

        Args:
            points (_type_, optional): 输入点云. Defaults to None.

        Returns:
            作为模型输入的点云和True, 表示使用了xyz坐标
        """
        if points is None:
            #* 选择哪几个维度作为输入参数，返回特征维度数
            num_output_features = len(self.used_feature_list)
            return num_output_features

        assert points.shape[-1] == len(self.src_feature_list)
        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True
