# C-CLOCs: Camera-LiDAR Object Candidates Fusion

CLOCs (Camera-LiDAR Object Candidates) 是一种相机-激光雷达后融合方法，将 2D 检测器（RetinaNet ResNet-18）和 3D 检测器（VoxelRCNN）的候选框通过融合网络（Fusion Head）进行联合推理，提升 3D 目标检测精度。

## 实验环境

- CUDA 11.8
- PyTorch 2.0.1
- Single RTX 4090
- Ubuntu 20.04
- MMDetection 3.0.0

## 数据准备

### 1. 拉取仓库

```bash
git clone https://github.com/zhangtingyu11/C-CLOCs.git
cd C-CLOCs
```

### 2. 准备 KITTI 数据集

```bash
cd data/
ln -s YOUR_KITTI_DATASET_PATH kitti
# 下载道路平面信息: https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view
```

KITTI 数据集目录结构：

```
├── ImageSets
├── testing
│   ├── calib
│   ├── image_2
│   └── velodyne
└── training
    ├── calib
    ├── image_2
    ├── label_2
    ├── planes
    └── velodyne
```

### 3. 通过 SAM 模型生成图像数据库

参考 [SAM 官方仓库](https://github.com/facebookresearch/segment-anything) 安装 SAM，下载 vit_h 权重到 `weights/` 目录。

```bash
python tools/generate_mask.py
```

### 4. 转换 KITTI 为 COCO 格式

```bash
python tools/kitti2coco.py
```

### 5. 安装 MMDetection

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
cd mmdetection && git checkout 3.0.0 && pip install -v -e .
cd ..
ln -s ../data mmdetection/data
```

### 6. 生成 KITTI 数据信息

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## 训练步骤

### Step 1: 训练 2D 检测器 (RetinaNet)

```bash
cd tools
python train_clocs.py
```

### Step 2: 训练 VoxelRCNN 3D 检测器基线

```bash
python train.py --cfg_file tools/cfgs/kitti_models/voxel_rcnn_car.yaml
```

### Step 3: 训练 C-CLOCs 融合网络

```bash
python tools/train_clocs_per_epoch_eval.py \
    --cfg_file tools/cfgs/kitti_models/voxel_rcnn_car_clocs_fusion_aug_v2.yaml \
    --extra_tag your_experiment_name \
    --pretrained_model output/kitti_models/voxel_rcnn_car/voxelrcnn_aug_baseline/ckpt/checkpoint_epoch_80.pth
```

**关键说明**：C-CLOCs 冻结 VoxelRCNN 的所有模块（VFE, BACKBONE_3D, BACKBONE_2D, DENSE_HEAD, ROI_HEAD），仅训练 Fusion Head。必须通过 `--pretrained_model` 加载预训练的 VoxelRCNN 权重。

## 最佳结果

### KITTI Car 3D Detection (AP_R40 @ 0.70)

| 模型 | Easy | Moderate | Hard |
|------|------|----------|------|
| VoxelRCNN (标准增强, 80 epochs) | 92.42 | 85.03 | 82.70 |
| C-CLOCs (fusion_gt_sampling + flip) | **92.82** | **86.04** | **83.23** |
| **提升** | **+0.40** | **+1.01** | **+0.53** |

CLOCs 融合在 Moderate 场景下带来了约 1 个点的稳定提升。

## 主要配置文件

| 文件 | 说明 |
|------|------|
| `tools/cfgs/kitti_models/voxel_rcnn_car_clocs_fusion_aug_v2.yaml` | CLOCs 主配置（含 fusion_gt_sampling 数据增强） |
| `tools/cfgs/kitti_models/voxel_rcnn_car_clocs_fusion_aug_v2_deep.yaml` | CLOCs 深度版本（更大融合网络） |
| `tools/cfgs/dataset_configs/database_generate_kitti.yaml` | 图像数据库生成配置 |
| `tools/cfgs/kitti_models/voxel_rcnn_car.yaml` | VoxelRCNN 基线配置 |

## 关键改进

相比原始 CLOCs 实现：
1. **数据增强**: 引入 `fusion_gt_sampling` 多模态 GT 采样，同时增强点云和图像
2. **融合网络**: 增加 ResBlock 跳跃连接和额外特征通道（USE_RES_BLOCK, USE_EXTRA_FEATURES）
3. **数据库质量**: IoU-based SAM mask 选择 + 自适应形态学闭运算 + 宽高比保持的复制粘贴
4. **训练策略**: Per-epoch evaluation with proper VoxelRCNN baseline comparison
