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
from pcdet.models.model_utils import model_nms_utils
from copy import deepcopy
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
        self.nhead = nhead
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
            nn.Linear(d_model, 8)  # Δx, Δy, Δz, Δl, Δw, Δh, Δθ
        )

    def forward(self, x_3d, x_2d, mask_3d=None, mask_2d=None):
        # x_3d: (B, N_3d, 8)
        # x_2d: (B, N_2d, 5)
        # mask_3d: (B, N_3d), True表示有效，False表示无效
        # mask_2d: (B, N_2d), True表示有效，False表示无效
        B, N_3d, _ = x_3d.shape
        B, N_2d, _ = x_2d.shape

        # 映射特征
        q = self.proj_3d(x_3d)  # (B, N_3d, d_model)
        k = v = self.proj_2d(x_2d)  # (B, N_2d, d_model)

        # --- 处理key_padding_mask ---
        key_padding_mask = ~mask_2d if mask_2d is not None else None  # (B, N_2d)

        # --- 处理attn_mask（关键修正）---
        attn_mask = None
        if mask_3d is not None and mask_2d is not None:
            # 1. 生成基础掩码 (B, N_3d, N_2d)
            attn_mask = torch.zeros((B, N_3d, N_2d), device=x_3d.device)  # 初始全0（允许关注）
            
            # 2. 填充无效位置为-inf
            for b in range(B):
                valid_3d = mask_3d[b]  # (N_3d,)
                valid_2d = mask_2d[b]  # (N_2d,)
                # 仅当3D和2D均有效时允许关注，否则屏蔽
                attn_mask[b, ~valid_3d, :] = float('-inf')  # 无效3D框屏蔽所有2D
                attn_mask[b, :, ~valid_2d] = float('-inf')  # 无效2D框被所有3D屏蔽

            # 3. 扩展为多头形状 (B * nhead, N_3d, N_2d)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N_3d, N_2d)
            attn_mask = attn_mask.expand(-1, self.nhead, -1, -1)  # (B, nhead, N_3d, N_2d)
            attn_mask = attn_mask.reshape(B * self.nhead, N_3d, N_2d)  # (B * nhead, N_3d, N_2d)

        # --- 注意力计算 ---
        attn_output, _ = self.cross_attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        corrections = self.correction(attn_output)
        return corrections


def load_cfg(cfg_file):
    """加载配置文件, 直接调用的官方的函数

    Args:
        cfg_file (_type_): 配置文件

    Returns:
        _type_: _description_
    """
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def build_dataset(cfg, root_path, logger, training):
    """构建数据集

    Args:
        cfg (_type_): 读取到的配置
        root_path (_type_): 数据集的根目录
        logger (_type_): 日志系统

    Returns:
        _type_: pytorch的Dataset类
    """
    dataset = KittiDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=training,
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
    train_dataset = build_dataset(cfg, "../data/kitti", logger, training=True)
    val_dataset = build_dataset(cfg, "../data/kitti", logger, training=False)
    # 加载模型
    model = build_model(cfg, train_dataset, "../ckpt/voxel_rcnn_car_84.54.pth", logger)
    
    # 构建修正模型
    correct_model = CrossAttention3DCorrector().cuda().train()
    
    # 定义优化器
    optimizer = torch.optim.Adam(correct_model.parameters(), lr=1e-4)
    
    # 定义训练轮数
    num_epochs = 10
    
    # 生成dataloader
    # TODO 后续需要支持多batch
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, pin_memory=True, num_workers=4,
        shuffle=True, collate_fn=train_dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0)
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, pin_memory=True, num_workers=4,
        shuffle=False, collate_fn=val_dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0)
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        correct_model.train()
        for iter_idx, data_dict in enumerate(tqdm(train_dataloader)):
            # 将数据加载到gpu上
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                # 先使用之前训练好的模型的到预测的3D包围框和得分
                pred_dicts, _ = model.forward(data_dict)
            pred_boxes_list = []
            pred_scores_list = []
            mask_3d_list = []
            for pred_dict in pred_dicts:
                mask_3d = torch.zeros_like(pred_dict["pred_scores"])
                mask_3d[:pred_dict["pred_scores"].shape[0]] = 1
                mask_3d_list.append(mask_3d.unsqueeze(0))
                if pred_dict["pred_boxes"].shape[0] < 100:
                    # 填充到100，并且填充的部分无梯度
                    pred_dict["pred_boxes"] = torch.cat([pred_dict["pred_boxes"], torch.zeros(100 - pred_dict["pred_boxes"].shape[0], 7).to(pred_dict["pred_boxes"].device)], dim=0)
                    pred_dict["pred_scores"] = torch.cat([pred_dict["pred_scores"], torch.zeros(100 - pred_dict["pred_scores"].shape[0]).to(pred_dict["pred_scores"].device)], dim=0)

                pred_boxes_list.append(pred_dict["pred_boxes"].unsqueeze(0))
                pred_scores_list.append(logit(pred_dict['pred_scores'].detach())[:, None].unsqueeze(0))
            
            mask_3d = torch.cat(mask_3d_list, dim=0)
            pred_boxes = torch.cat(pred_boxes_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)
            
            # 将包围框和得分拼接起来， (N, 7)
            boxes_3d = torch.cat([pred_boxes, pred_scores], dim=-1)
            
            # 获取2D gt 包围框
            bboxes_2d = data_dict["gt_boxes2d"]
            mask_2d = (bboxes_2d[:, :, 3] != 0).cuda()
            image_height = data_dict["image_shape"][:, 0]
            image_width = data_dict["image_shape"][:, 1]
            # 随机分数在0.9~1
            random_scores = torch.ones((bboxes_2d.shape[0], bboxes_2d.shape[1], 1), dtype=bboxes_2d.dtype, device=bboxes_2d.device)
            bboxes_2d = torch.cat([bboxes_2d, random_scores], dim=-1)
                    
            # bboxes_2d归一化到0-1
            bboxes_2d[..., 0] = bboxes_2d[..., 0] / image_width.unsqueeze(-1)
            bboxes_2d[..., 1] = bboxes_2d[..., 1] / image_height.unsqueeze(-1)
            bboxes_2d[..., 2] = bboxes_2d[..., 2] / image_width.unsqueeze(-1)
            bboxes_2d[..., 3] = bboxes_2d[..., 3] / image_height.unsqueeze(-1)
            
            corrections = correct_model(boxes_3d.float().cuda(), bboxes_2d.float().cuda(), mask_2d)
            
            # 在assign_targets的时候需要输入字典里面有rois, roi_scores, roi_labels
            # 这边的目标应该是以原来3D模型的输出来的
            # TODO 需要确认roi_labels是否应该设置成全1
            data_dict["rois"] = boxes_3d[:, :, 0:7]
            data_dict["roi_scores"] = boxes_3d[:, :, 7]
            data_dict["roi_labels"] = torch.ones_like(data_dict["roi_scores"]).long()
            data_dict["no_sample"] = True
            # TODO assign_targets有问题
            target_dict = model.roi_head.assign_targets(data_dict)
            
            
            target_dict["rcnn_reg"] = corrections[:, :, :7]
            target_dict["rcnn_cls"] = target_dict["roi_scores"] + corrections[:, :, 7]
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
        avg_loss = running_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {avg_loss}")
        
        # 每个epoch结束后保存模型
        torch.save(correct_model.state_dict(), f"correct_model_epoch_{epoch + 1}.pth")

        # 验证集验证
        correct_model.eval()
        # 存储最后的预测框
        det_annos = []
        class_names = val_dataset.class_names
        for iter_idx, data_dict in enumerate(tqdm(val_dataloader)):
            # 将数据加载到gpu上
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                # 先使用之前训练好的模型的到预测的3D包围框和得分
                pred_dicts, _ = model.forward(data_dict)
            pred_boxes_list = []
            pred_scores_list = []
            mask_3d_list = []
            for pred_dict in pred_dicts:
                mask_3d = torch.zeros_like(pred_dict["pred_scores"])
                mask_3d[:pred_dict["pred_scores"].shape[0]] = 1
                mask_3d_list.append(mask_3d.unsqueeze(0))
                if pred_dict["pred_boxes"].shape[0] < 100:
                    # 填充到100，并且填充的部分无梯度
                    pred_dict["pred_boxes"] = torch.cat([pred_dict["pred_boxes"], torch.zeros(100 - pred_dict["pred_boxes"].shape[0], 7).to(pred_dict["pred_boxes"].device)], dim=0)
                    pred_dict["pred_scores"] = torch.cat([pred_dict["pred_scores"], torch.zeros(100 - pred_dict["pred_scores"].shape[0]).to(pred_dict["pred_scores"].device)], dim=0)

                pred_boxes_list.append(pred_dict["pred_boxes"].unsqueeze(0))
                pred_scores_list.append(logit(pred_dict['pred_scores'].detach())[:, None].unsqueeze(0))
            
            mask_3d = torch.cat(mask_3d_list, dim=0)
            pred_boxes = torch.cat(pred_boxes_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)
            
            # 将包围框和得分拼接起来， (N, 7)
            boxes_3d = torch.cat([pred_boxes, pred_scores], dim=-1)
            
            # 获取2D gt 包围框
            bboxes_2d = data_dict["gt_boxes2d"]
            mask_2d = (bboxes_2d[:, :, 3] != 0).cuda()
            image_height = data_dict["image_shape"][:, 0]
            image_width = data_dict["image_shape"][:, 1]
            # 随机分数在0.9~1
            random_scores = torch.ones((bboxes_2d.shape[0], bboxes_2d.shape[1], 1), dtype=bboxes_2d.dtype, device=bboxes_2d.device)
            bboxes_2d = torch.cat([bboxes_2d, random_scores], dim=-1)
                    
            # bboxes_2d归一化到0-1
            bboxes_2d[..., 0] = bboxes_2d[..., 0] / image_width.unsqueeze(-1)
            bboxes_2d[..., 1] = bboxes_2d[..., 1] / image_height.unsqueeze(-1)
            bboxes_2d[..., 2] = bboxes_2d[..., 2] / image_width.unsqueeze(-1)
            bboxes_2d[..., 3] = bboxes_2d[..., 3] / image_height.unsqueeze(-1)
            
            with torch.no_grad():
                corrections = correct_model(boxes_3d.float().cuda(), bboxes_2d.float().cuda(), mask_2d)
            
            # 构建最终的预测框
            pred_boxes = boxes_3d[:, :, :7] + corrections[:, :, :7]
            pred_scores = boxes_3d[:, :, 7] + corrections[:, :, 7]
            
            # NMS操作            
            cls_preds, label_preds = torch.max(pred_scores, dim=-1)
            label_preds = label_preds + 1 
            post_process_cfg = deepcopy(cfg.MODEL.POST_PROCESSING)
            post_process_cfg.SCORE_THRESH=0.3
            post_process_cfg.NMS_CONFIG.NMS_THRESH=0.1
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=torch.sigmoid(cls_preds), box_preds=pred_boxes,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            # if post_process_cfg.OUTPUT_RAW_SCORE:
            #     max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
            #     selected_scores = max_cls_preds[selected]

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = pred_boxes[selected]
            
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            res_dict = []
            res_dict.append(record_dict)
            annos = val_dataset.generate_prediction_dicts(
                data_dict, pred_dicts, class_names,
                output_path=None
            )
            det_annos += annos

        result_str, result_dict = val_dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=None
        )
        logger.info(result_str)
        
if __name__ == "__main__":
    main()
        
        
        

if __name__ == "__main__":
    main()
        
    