from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        #* [batch_size, N, 3]->[batch_size, 3, N]
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.farthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz: 是否使用xyz数据
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

class _PointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.radii = []

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None
        self.confidence_mlp = None

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None,
                counts=None,):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        #* xyz_flipped： [batch_size, 每个sample中的点数, 3] -> [batch_size, 3, 每个sample中的点数]
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                #* 在sample_range_list[i]的范围内进行采样，将采样范围内的点先取出来
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':
                    #* 如果使用D-FPS，就调用下面的代码，
                    #* 输入参数：
                    #*      第一个:采样范围内的点
                    #*      第二个:采样的点的个数
                    #* 输出参数:
                    #*      sample_idx:采样的点的索引，[batch_size, 需要采样的点数]
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )
                elif self.sample_method_list[i] == 'ds-fps':
                    assert counts is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    counts_slice = \
                        counts[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    if(self.use_kde_count):
                        scores_slice = torch.mul(scores_slice, (torch.exp(-counts_slice)) ** self.weight_lambda)
                    else:
                        if(self.use_density_sigmoid):
                            if(self.positive_corr):
                                scores_slice = torch.mul(scores_slice, (torch.sigmoid(torch.log10(counts_slice))) ** self.weight_lambda)
                            else:
                                scores_slice = torch.mul(scores_slice, (1-torch.sigmoid(torch.log10(counts_slice))) ** self.weight_lambda)
                        else:
                            scores_slice = torch.mul(scores_slice, (torch.exp(-torch.log10(counts_slice+1))) ** self.weight_lambda)
                    
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )
                elif self.sample_method_list[i] == 'df-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    counts_slice = counts[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    density = torch.sigmoid(torch.log10(counts_slice)).unsqueeze(1)
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling_with_density(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                density,
                                                                                self.weight_gamma,
                                                                                self.weight_alpha)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                else:
                    raise NotImplementedError
                #* 采样点的索引需要加上采样范围的起始点的索引
                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])
            #* 将多次采样点的索引拼接起来
            sample_idx = torch.cat(sample_idx_list, dim=-1)
            #* 输入参数:
            #*      xyz_flipped:采样范围内的点,[batch_size, 3, 采样范围内的点数]
            #*      sample_idx: 采样的点的索引,[batch_size, 需要采样的点数]
            #* 输出参数:
            #*      new_xyz:[batch_size, 需要采样的点数, 3]
            #* 这边就是简单的把采样的点的坐标求出来，不涉及什么MLP，MAXPOOLING
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)
            #* 获取采样点的特征
            if self.skip_connection: 
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)
        idx_cnt = None
        for i in range(len(self.groupers)):
            #* grouper的输入参数
            #*      xyz: 采样范围内的点坐标,[batch_size, 采样范围内的点数, 3]
            #*      new_xyz: 采样的点的坐标, [batch_size, 需要采样的点的个数, 3]
            #*      features: 采样范围内的点特征,[batch_size, 点的特征长度, 采样范围内的点数]
            #* 输出参数:
            #*      idx_cnt: 每个采样点group内的真实点个数
            #*      new_features: group后的特征，[batch_size, 特征维度, 采样的点个数， group中的最大点个数]
            # idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features, cur_idx_cnt = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            if(isinstance(self.groupers[i], pointnet2_utils.QueryAndGroupDilated)):
                if(idx_cnt is None):
                    idx_cnt = cur_idx_cnt
                else:
                    idx_cnt += cur_idx_cnt
            else:
                idx_cnt = cur_idx_cnt

            #* 使用1*1卷积来处理特征，当做MLP
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # #* 选取group中点数 > 0的采样点, [batch_size, 采样点个数]
            # idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
            # #* idx_cnt_mask: [batch_size, 采样点个数] -> [batch_size, 1, 采样点个数, 1]
            # idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
            # #* group中个数是0，那特征就是0
            # new_features *= idx_cnt_mask
            #* pooling操作， 最后得到的pooled_features为[batch_size, 特征维度, 采样点个数， 1]
            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            #* 将这个ball query后的pooled_features([batch_size, 特征维度, 采样点个数])加进new_features_list中
            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)
        #* 将ball_query前的采样点特征也加进new_features_list中
        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)
        #* 将ball query前的特征和不同半径下ball query后的特征进行拼接，得到采样点的特征
        new_features = torch.cat(new_features_list, dim=1)
        #* self.aggregation_mlp是1*1卷积进行特征降维+ BN+ Relu
        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        if self.confidence_mlp is not None:
            new_scores = self.confidence_mlp(new_features)
            new_scores = new_scores.squeeze(1)  # (B, npoint)
            return new_xyz, new_features, new_scores, idx_cnt
        #* 返回采样点的坐标， 采样点的特征，None, 
        return new_xyz, new_features, None, idx_cnt


class PointnetSAModuleFSMSG(_PointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 fusion_type: str = 'concatation', 
                 bn: bool = True,
                 use_xyz: bool = True,
                 use_density: bool = False,
                 use_kde: bool = False,
                 use_kde_count: bool = False,
                 use_distance_to_center: bool = False,
                 use_distance_to_origin: bool = False,
                 use_relative_direction_angle: bool = False,
                 use_absolute_direction_angle: bool = False,
                 use_sincos: bool = False,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 use_density_sigmoid: bool = False,
                 weight_gamma: float = 1.0,
                 weight_lambda: float = 1.0,
                 weight_alpha: float = 1.0, 
                 positive_corr: bool = False,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None,
                 extra_dim_mlp: List[int] = None,
                ):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param fusion_type: 特征融合方式, 可选相加或者拼接
        :param use_xyz:
        :param use_density:
        :param use_distance_to_center:
        :param use_relative_direction_angle:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param extra_dim_mlp:list of int, spec extra dim mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.radii = radii
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.use_kde_count  = use_kde_count
        self.use_density_sigmoid = use_density_sigmoid

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            cur_extra_dim_mlp = None
            if(extra_dim_mlp is not None):
                cur_extra_dim_mlp = extra_dim_mlp[i]
            if dilated_radius_group:
                #* QueryAndGroupDilated会涉及两个参数:
                #* former_radius:之前的采样半径
                #* radius: 现在的采样半径
                #*      然后在former_radius和radius之间进行采样
                #* nsample: 每个group中点的最大个数
                #* use_xyz：是否使用点的坐标
                self.groupers.append(
                    pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz,
                                                         use_density = use_density, 
                                                         use_distance_to_center = use_distance_to_center,
                                                         use_distance_to_origin = use_distance_to_origin,
                                                         use_relative_direction_angle = use_relative_direction_angle,
                                                         use_absolute_direction_angle=use_absolute_direction_angle,
                                                         use_sincos=use_sincos,
                                                         extra_dim_mlp = cur_extra_dim_mlp,
                                                         fusion_type = fusion_type,
                    )
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz,
                                                         use_density = use_density, 
                                                         use_kde = use_kde,
                                                         use_kde_count = use_kde_count,
                                                         use_distance_to_center = use_distance_to_center,
                                                         use_distance_to_origin = use_distance_to_origin,
                                                         use_relative_direction_angle = use_relative_direction_angle,
                                                         use_absolute_direction_angle=use_absolute_direction_angle,
                                                         use_sincos=use_sincos,
                                                         extra_dim_mlp = cur_extra_dim_mlp,
                                                         fusion_type = fusion_type,
                    )
                )
            former_radius = radius
            mlp_spec = mlps[i]
            if(fusion_type == "concatation"):
                if extra_dim_mlp:
                    mlp_spec[0] += cur_extra_dim_mlp[-1]
                else:
                    if use_xyz:
                        mlp_spec[0] += 3
                    if use_density:
                        mlp_spec[0] += 1
                    if use_kde:
                        mlp_spec[0] += 1
                    if use_relative_direction_angle:
                        if use_sincos:
                            mlp_spec[0] += 6
                        else:
                            mlp_spec[0] += 3
                    if use_absolute_direction_angle:
                        if use_sincos:
                            mlp_spec[0] += 6
                        else:
                            mlp_spec[0] += 3
                    if use_distance_to_center:
                        mlp_spec[0] += 1
                    if use_distance_to_origin:
                        mlp_spec[0] += 1

            shared_mlp = []
            for k in range(len(mlp_spec) - 1):
                shared_mlp.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlp))
            in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma
        self.weight_lambda = weight_lambda
        self.weight_alpha = weight_alpha
        self.positive_corr = positive_corr

        if skip_connection:
            out_channels += in_channels

        if aggregation_mlp is not None:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None

        if confidence_mlp is not None:
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, 1, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
        else:
            self.confidence_mlp = None


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
