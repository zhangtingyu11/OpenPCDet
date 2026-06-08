"""
Generate KITTI GT database for VoxelRCNN training.
Run from tools/ directory after adding parent dir to path.
"""
import sys
sys.path.insert(0, '../')
from pathlib import Path
import yaml
from easydict import EasyDict
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

root_path = Path('../data/kitti')

# Load dataset config with DATA_AUGMENTOR etc.
with open('cfgs/dataset_configs/kitti_dataset.yaml', 'r') as f:
    dataset_cfg = EasyDict(yaml.safe_load(f))

# Set required attributes
dataset_cfg.INFO_PATH = {'train': [root_path / 'kitti_infos_train.pkl']}
dataset_cfg.POINT_CLOUD_RANGE = [0, -40, -3, 70.4, 40, 1]
dataset_cfg.DATA_PATH = str(root_path)

ds = KittiDataset(
    dataset_cfg=dataset_cfg,
    class_names=['Car'],
    root_path=root_path,
    training=True
)

print("Generating GT database...")
ds.create_groundtruth_database(
    info_path=str(root_path / 'kitti_infos_train.pkl'),
    used_classes=['Car'],
    split='train'
)
print("Done!")
