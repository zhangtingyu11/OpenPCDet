import torch
import glob
import pickle
import numpy as np
import torch.nn.functional as F
from pcdet.datasets import DatasetTemplate, KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pathlib import Path
from pcdet.utils import calibration_kitti, common_utils
from pcdet.models import build_network, load_data_to_gpu
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pcdet.utils import loss_utils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = common_utils.create_logger()

class CrossAttention3DCorrector(nn.Module):
    def __init__(self, d_3d=8, d_2d=5, d_model=128, nhead=4):
        """定义一个修正模型

        Args:
            d_3d (int, optional): 初始的3D包围框的维度. Defaults to 8.
            d_2d (int, optional): 初始的2D包围框的维度. Defaults to 5.
            d_model (int, optional): 需要扩展到的隐藏层的维度. Defaults to 128.
            nhead (int, optional): 注意力的头数. Defaults to 4.
        """
        super().__init__()
        # 3D特征的投影层
        self.proj_3d = nn.Sequential(
            nn.Linear(d_3d, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # 2D特征的投影层
        self.proj_2d = nn.Sequential(
            nn.Linear(d_2d, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # 计算完交叉注意力之后的映射层
        self.correction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 7)  # Δx, Δy, Δz, Δl, Δw, Δh, Δθ
        )

    def forward(self, x_3d, x_2d):
        # x_3d: (B, 100, 8)
        # x_2d: (B, N, 5)
        B, N_3d, _ = x_3d.shape
        B, N_2d, _ = x_2d.shape

        # 映射特征
        q = self.proj_3d(x_3d)  # (B, 100, d_model)
        k = v = self.proj_2d(x_2d)  # (B, N, d_model)

        attn_output, _ = self.cross_attn(q, k, v)  # (B, 100, d_model)
        corrections = self.correction(attn_output)  # (B, 100, 7)

        # 对前面的维度进行修正，但是得分并不修正
        corrected_3d = x_3d.clone()
        corrected_3d[..., :7] += corrections  # x, y, z, l, w, h, θ += Δ

        return corrected_3d


def load_cfg(cfg_file):
    """加载配置文件, 直接调用的官方的函数

    Args:
        cfg_file (_type_): 配置文件

    Returns:
        _type_: _description_
    """
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def build_dataset(cfg, root_path, logger):
    """构建数据集

    Args:
        cfg (_type_): 读取到的配置
        root_path (_type_): 数据集的根目录
        logger (_type_): 日志系统

    Returns:
        _type_: pytorch的Dataset类
    """
    dataset = KittiDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=True,
            root_path=Path(root_path), logger=logger
        )
    return dataset

def build_model(cfg, dataset, ckpt_file, logger):
    """构建模型

    Args:
        cfg (_type_): 读取到的配置
        dataset (_type_): 数据集
        ckpt_file (_type_): 使用的ckpt文件
        logger (_type_): 日志系统

    Returns:
        _type_: 构造好的模型
    """
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=True)
    model.cuda().eval()
    return model

def load_pickle_file(pickle_file):
    """加载生成的pickle文件, 构建一个frame_id到data的字典

    Args:
        pickle_file (_type_): pickle文件

    Returns:
        _type_: _description_
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
        data_dict = {
            item['point_cloud']['lidar_idx']: item for item in data
        }
    return data_dict


def get_cls_loss(forward_ret_dict, cls_weight=1.0):
    """获取分类loss

    Args:
        pred_scores (_type_): _description_
        gt_scores (_type_): _description_
        loss_weight (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    rcnn_cls = forward_ret_dict['rcnn_cls']
    rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
    rcnn_cls_flat = rcnn_cls.view(-1)
    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
    cls_valid_mask = (rcnn_cls_labels >= 0).float()
    rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

    rcnn_loss_cls = rcnn_loss_cls * cls_weight
    tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
    return rcnn_loss_cls, tb_dict

def get_box_loss(forward_ret_dict, model, loss_weight=1.0):
    smooth_l1_loss = loss_utils.WeightedSmoothL1Loss(code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    box_coder = model.roi_head.box_coder
    code_size = box_coder.code_size
    reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
    gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
    gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
    rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
    roi_boxes3d = forward_ret_dict['rois']
    rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

    fg_mask = (reg_valid_mask > 0)
    fg_sum = fg_mask.long().sum().item()

    tb_dict = {}

    rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
    rois_anchor[:, 0:3] = 0
    rois_anchor[:, 6] = 0
    reg_targets = box_coder.encode_torch(
        gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
    )

    rcnn_loss_reg = smooth_l1_loss(
        rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
        reg_targets.unsqueeze(dim=0),
    )  # [B, M, 7]
    rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
    rcnn_loss_reg = rcnn_loss_reg * loss_weight
    tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

    if fg_sum > 0:
        # TODO: NEED to BE CHECK
        fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
        fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

        fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
        batch_anchors = fg_roi_boxes3d.clone().detach()
        roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
        roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
        batch_anchors[:, :, 0:3] = 0
        rcnn_boxes3d = box_coder.decode_torch(
            fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
        ).view(-1, code_size)

        rcnn_boxes3d = common_utils.rotate_points_along_z(
            rcnn_boxes3d.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        rcnn_boxes3d[:, 0:3] += roi_xyz

        loss_corner = loss_utils.get_corner_loss_lidar(
            rcnn_boxes3d[:, 0:7],
            gt_of_rois_src[fg_mask][:, 0:7]
        )
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * loss_weight

        rcnn_loss_reg += loss_corner
        tb_dict['rcnn_loss_corner'] = loss_corner.item()

    return rcnn_loss_reg, tb_dict

def logit(p):
    """sigmoid的反函数

    Args:
        p (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.special.logit(p)

def main():
    # 加载配置文件
    cfg = load_cfg("cfgs/kitti_models/voxel_rcnn_car_without_nms.yaml")
    # 加载数据集
    dataset = build_dataset(cfg, "../data/kitti", logger)
    # 加载模型
    model = build_model(cfg, dataset, "../ckpt/voxel_rcnn_car_84.54.pth", logger)
    
    # 构建修正模型
    correct_model = CrossAttention3DCorrector().cuda().train()
    
    # 定义优化器
    optimizer = torch.optim.Adam(correct_model.parameters(), lr=1e-4)
    
    # 定义训练轮数
    num_epochs = 10
    
    # 生成dataloader
    # TODO 后续需要支持多batch
    dataloader = DataLoader(
        dataset, batch_size=1, pin_memory=True, num_workers=4,
        shuffle=True, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0)
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        for iter_idx, data_dict in enumerate(tqdm(dataloader)):
            # 将数据加载到gpu上
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                # 先使用之前训练好的模型的到预测的3D包围框和得分
                pred_dicts, _ = model.forward(data_dict)
                pred_boxes = pred_dicts[0]['pred_boxes'].detach()
                pred_scores = logit(pred_dicts[0]['pred_scores'].detach())
            
            # 将包围框和得分拼接起来， (N, 7)
            boxes_3d = torch.cat([pred_boxes, pred_scores[:, None]], dim=1).unsqueeze(0)
            
            # 获取2D gt 包围框
            bboxes_2d = data_dict["gt_boxes2d"]
            image_height = data_dict["image_shape"][:, 0]
            image_width = data_dict["image_shape"][:, 1]
            # 随机分数在0.9~1
            random_scores = torch.FloatTensor(bboxes_2d.shape[1], 1).uniform_(0.9, 1).cuda().unsqueeze(0)
            bboxes_2d = torch.cat([bboxes_2d, random_scores], dim=-1)
                    
            # bboxes_2d归一化到0-1
            bboxes_2d[..., 0] = bboxes_2d[..., 0] / image_width
            bboxes_2d[..., 1] = bboxes_2d[..., 1] / image_height
            bboxes_2d[..., 2] = bboxes_2d[..., 2] / image_width
            bboxes_2d[..., 3] = bboxes_2d[..., 3] / image_height
            
            correct_boxes_3d = correct_model(boxes_3d.float().cuda(), bboxes_2d.float().cuda())
            
            # 在assign_targets的时候需要输入字典里面有rois, roi_scores, roi_labels
            # TODO 需要确认roi_labels是否应该设置成全1
            data_dict["rois"] = correct_boxes_3d[:, :, 0:7]
            data_dict["roi_scores"] = correct_boxes_3d[:, :, 7]
            data_dict["roi_labels"] = torch.ones_like(data_dict["roi_scores"]).long()
            model.roi_head.train()
            target_dict = model.roi_head.assign_targets(data_dict)
            
            
            target_dict["rcnn_reg"] = target_dict["rois"]
            target_dict["rcnn_cls"] = target_dict["roi_scores"]
            cls_loss, _ = get_cls_loss(target_dict)
            bbox_loss, _= get_box_loss(target_dict, model)
            
            
            loss = cls_loss + bbox_loss
            if iter_idx%100 == 0:
                logger.info("cls_loss: {}, bbox_loss: {}, loss: {}".format(cls_loss.item(), bbox_loss.item(), loss.item()))
            
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
            
            running_loss += loss.item()
        
        # 每个epoch结束后打印平均损失
        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {avg_loss}")
        
        # 每个epoch结束后保存模型
        torch.save(correct_model.state_dict(), f"correct_model_epoch_{epoch + 1}.pth")

if __name__ == "__main__":
    main()
        
        
        

if __name__ == "__main__":
    main()
        
    