# C-CLOCs: Contrastive Camera-LiDAR Object Candidates Fusion

CLOCs (Camera-LiDAR Object Candidates) 是一种相机-激光雷达后融合方法，将 2D 检测器和 3D 检测器的候选框通过融合网络进行联合推理。

C-CLOCs 在原始 CLOCs 基础上引入对比学习，迫使融合网络学习匹配/不匹配跨模态候选对之间的特征一致性，并采用多模态 GT 采样 (MGS) 进行数据增强。

论文: *Contrastive Camera-LiDAR Object Candidates* (IEEE TIV, 2025)

## 环境配置

```bash
# 基础环境
CUDA 11.8
PyTorch 2.0.1
Python 3.8
GPU: RTX 4090 (24GB)
OS: Ubuntu 20.04

# 创建 conda 环境
conda create -n openpcdet_cu11.8 python=3.8 -y
conda activate openpcdet_cu11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安装 OpenPCDet
git clone https://github.com/zhangtingyu11/C-CLOCs.git
cd C-CLOCs
pip install -r requirements.txt
python setup.py develop

# 安装 SAM (Segment Anything Model)
pip install git+https://github.com/facebookresearch/segment-anything.git

# 安装 MMDetection 3.0.0
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
cd mmdetection
git checkout 3.0.0
pip install -v -e .
cd ..

# 安装其他依赖
pip install opencv-python numba tensorboardX tqdm pyyaml
```

## 数据准备

### 1. 下载 KITTI 数据集

从 [KITTI 官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 下载以下文件：
- `data_object_image_2.zip` (left color images)
- `data_object_velodyne.zip` (Velodyne point clouds)
- `data_object_label_2.zip` (training labels)
- `data_object_calib.zip` (calibration files)

下载 [道路平面信息](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view) 并解压到 `data/kitti/training/planes/`。

下载 [ImageSets](https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets) 到 `data/kitti/ImageSets/`。

目录结构应为：

```
data/kitti/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   ├── trainval.txt
│   └── test.txt
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   ├── planes/
│   └── velodyne/
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/
```

### 2. 下载 SAM 权重

```bash
mkdir -p weights
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

### 3. 生成图像数据库

使用 SAM vit_h 对 KITTI 训练图像进行实例分割，生成多模态 GT 数据库：

```bash
python tools/generate_mask.py --cfg_file tools/cfgs/dataset_configs/database_generate_kitti.yaml
```

该脚本会：
1. 用 SAM vit_h 对每张训练图像做实例分割
2. 用 IoU 匹配选择最佳 mask（阈值 0.3）
3. 自适应形态学闭运算细化 mask 边界
4. 按角度-距离网格分箱存储样本
5. 输出 `data/kitti/image_database_train.pkl`

### 4. 转换 KITTI 为 COCO 格式

```bash
python tools/kitti2coco.py
```

生成的数据结构：

```
data/
├── kitti/
│   ├── image_gt_database_train/
│   ├── image_database_train.pkl
│   └── ...
└── coco/
    ├── annotations/
    ├── coco_label_2/
    ├── labels/
    │   ├── train_labels/
    │   └── val_labels/
    ├── train2017/
    └── val2017/
```

### 5. 软链接 mmdetection 数据目录

```bash
cd mmdetection
ln -sf ../data data
cd ..
```

### 6. 生成 KITTI 数据信息文件

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## 训练步骤

### Step 1: 训练 2D 检测器 (RetinaNet ResNet-18)

```bash
cd tools
python train_clocs.py
cd ..
```

该脚本会自动：
1. 从 MMDetection 3.0.0 model zoo 下载 RetinaNet R18 FPN 1x 配置和权重
2. 将类别修改为 KITTI 单类 (Car)
3. 用 8 卡模拟的 batch size 训练 12 epochs
4. 输出到 `mmdetection/work_dirs_single_class/retinanet_r18_fpn_1x_coco/`
5. 最终权重: `mmdetection/work_dirs_single_class/retinanet_r18_fpn_1x_coco/epoch_9.pth`

### Step 2: 训练 VoxelRCNN 3D 检测器基线

```bash
python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/voxel_rcnn_car.yaml \
    --extra_tag voxelrcnn_aug_baseline
```

关键配置：
- 80 epochs, batch_size=16
- Adam OneCycle, LR=0.01
- GT sampling + random flip/rotation/scaling 数据增强
- 输出: `output/kitti_models/voxel_rcnn_car/voxelrcnn_aug_baseline/`
- 最终权重: `output/kitti_models/voxel_rcnn_car/voxelrcnn_aug_baseline/ckpt/checkpoint_epoch_80.pth`

### Step 3: 训练 C-CLOCs 融合网络

```bash
python tools/train_clocs_per_epoch_eval.py \
    --cfg_file tools/cfgs/kitti_models/voxel_rcnn_car_clocs_fusion_aug_v2.yaml \
    --extra_tag your_experiment_name \
    --pretrained_model output/kitti_models/voxel_rcnn_car/voxelrcnn_aug_baseline/ckpt/checkpoint_epoch_80.pth
```

**关键说明**：
- C-CLOCs 冻结 VoxelRCNN 的所有模块（VFE, Backbone_3D, Backbone_2D, Dense_Head, ROI_Head），仅训练 Fusion Head
- **必须**通过 `--pretrained_model` 加载预训练的 VoxelRCNN 权重，否则冻结模块处于随机初始化状态，融合头无法学到有效信息
- 训练 3 epochs
- 每个 epoch 结束后自动在 val 集上评估

**配置文件说明**：

| 配置文件 | 融合网络配置 |
|---------|------------|
| `voxel_rcnn_car_clocs_fusion_aug.yaml` | 基础版: NUM_FILTERS=[18,36,36,1], 无 ResBlock, 无 ExtraFeat |
| `voxel_rcnn_car_clocs_fusion_aug_v2.yaml` | 增强版: NUM_FILTERS=[18,36,36,1], ResBlock=True, ExtraFeat=True |
| `voxel_rcnn_car_clocs_fusion_aug_v2_deep.yaml` | 深度版: NUM_FILTERS=[32,64,128,64,1], ResBlock=True, ExtraFeat=True |

主要配置差异来自 `MODEL.FUSION_HEAD`:
- `USE_RES_BLOCK`: 是否在卷积层间使用跳跃连接
- `USE_EXTRA_FEATURES`: 是否融合额外的 LiDAR 特征通道
- `NUM_FILTERS`: 融合网络各层通道数（值越大网络越深）
- `USE_CONTRA`: 是否使用对比学习
- `USE_LA`: 是否使用标签分配策略
- `CLOCS_SCORE_WEIGHT`: CLOCs 分数与 3D 分数的融合权重

## 最佳结果

### KITTI Car 3D Detection (AP_R40 @ 0.70, val set)

训练方案：train 集训练，val 集评估

| 模型 | Easy | Moderate | Hard |
|------|------|----------|------|
| VoxelRCNN (标准增强, 80 epochs) | 92.42 | 85.03 | 82.70 |
| C-CLOCs (v2 配置, fusion_gt_sampling + flip) | **92.82** | **86.04** | **83.29** |
| **提升** | **+0.40** | **+1.01** | **+0.59** |

C-CLOCs 最佳结果使用 `voxel_rcnn_car_clocs_fusion_aug_v2.yaml` 配置，在 epoch 1 取得。

### 消融实验结果

| 配置 | ResBlock | ExtraFeat | Easy | Moderate | Hard |
|------|----------|-----------|------|----------|------|
| 基础 (contra_la) | ✗ | ✗ | 92.65 | 85.71 | 83.18 |
| +ExtraFeat | ✗ | ✓ | 92.74 | 85.85 | 83.05 |
| +ResBlock | ✓ | ✗ | 92.72 | 85.85 | 83.11 |
| +fusion_gt_sampling | ✓ | ✗ | 92.82 | 85.93 | 83.16 |
| +fusion_gt_sampling (v2) | ✓ | ✓ | **92.82** | **86.04** | 83.23 |
| +fusion_gt_sampling (v2_deep) | ✓ | ✓ | 92.92 | 86.00 | **83.29** |

## 预测与提交

### 在 test 集上预测

```bash
cd tools
# 修改 test.py 中的 cfg_file 和 ckpt 路径
python test.py
```

### 提交 KITTI 测试服务器

```bash
# 预测结果在 output/ 对应目录下
# 将结果打包提交到 KITTI 官网评测服务器
```

## 代码改进记录

相比原始 OpenPCDet + CLOCs 实现：

1. **图像数据库生成** (`tools/generate_mask.py`):
   - IoU-based SAM mask 选择（替代脆弱的中心点选择）
   - 自适应形态学闭运算核大小（根据物体图像尺寸分 3 档）
   - 过滤严重遮挡（occlusion > 1）和截断（truncation >= 0.5）的样本

2. **多模态复制粘贴** (`pcdet/datasets/augmentor/database_sampler.py`):
   - 宽高比保持的图像缩放粘贴（替代直接拉伸）
   - 修复 RGB 回退 mask 始终为 True 的 bug

3. **融合网络** (`pcdet/models/fusion_heads/clocs_voxelrcnn_head.py`):
   - ResBlock 跳跃连接（USE_RES_BLOCK）
   - 额外 LiDAR 特征通道（USE_EXTRA_FEATURES）

4. **训练工具** (`tools/train_clocs_per_epoch_eval.py`):
   - Per-epoch 自动评估
   - 以真实 VoxelRCNN 结果为基线进行对比

## 引用

如果使用此代码，请引用：

```bibtex
@article{zhang2025cclocs,
  title={Contrastive Camera-LiDAR Object Candidates},
  author={Zhang, Tingyu and Liang, Zhigang and Yang, Yanzhao and Yang, Xinyu and Zhu, Yu and Wang, Jian},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2025},
  volume={10},
  number={5},
  pages={3442--3457}
}
```
