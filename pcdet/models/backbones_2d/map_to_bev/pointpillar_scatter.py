import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        """_summary_

        Args:
            model_cfg (_type_): 模型配置
                NUM_BEV_FEATURES: 表示特征维度
            grid_size (_type_): xyz方向voxel的个数, [nx, ny, nz]
        """
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """ 
        Args:
            batch_dict (_type_): _description_
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

        Returns:
            _type_: _description_
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            #* 挑选当前batch_idx的voxe坐标, [当前batch_idx的非空voxel个数, 4]
            this_coords = coords[batch_mask, :]
            #* this_coords存放的是[batch_idx, zidx, yidx, xidx], 因为zidx一定是0, 其实可以忽略
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            #* spatial_feature[:, indices]的shape是当前batch_idx的特征维度的 * 非空voxel个数的, 因此需要转置
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        #* batch_size * 特征维度 * (nx*ny*nz)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        #* batch_size * 特征维度 * (nx*ny*nz) -> batch_size * (特征维度*nz) * (nx*ny)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
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
            spatial_features: BEV特征图, batch_size * 特征维度 * H * W
        """
        return batch_dict
