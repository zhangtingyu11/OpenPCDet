import pickle
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm
import imageio as io
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import box_utils


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)


    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()

        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class DepthWeightGenerator:
    def __init__(self, pickle_file, class_names=['Car', 'Pedestrian', 'Cyclist'], radius = 30, data_root = None, config_file = None):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            self.data = {}
            for item in data:
                frame_id = item["image"]["image_idx"]
                self.data[frame_id] = item
        cfg_from_yaml_file(config_file, cfg)
        self.cfg = cfg
        self.class_names = class_names
        self.radius = radius
        self.data_root = Path(data_root)
        
    def check_in_range(self, bbox3d):
        limit_range = self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        if bbox3d.shape[0] > 7:
            bbox3d = bbox3d[0:7]
        box_centers = bbox3d[0:3]
        if ((box_centers >= limit_range[0:3]) & (box_centers <= limit_range[3:6])).all():
            return True
        else:
            return False
    
    def get_centers_2d(self, frame_id):
        centers_2d = []
        frame_id = str(frame_id).zfill(6)
        data = self.data[frame_id]
        annos = data['annos']
        for idx, class_name in enumerate(annos["name"]):
            if class_name not in self.class_names:
                continue
            bbox2d = annos["bbox"][idx]
            bbox3d = annos["gt_boxes_lidar"][idx]
            if not self.check_in_range(bbox3d):
                continue
            center_x, center_y = (bbox2d[0]+bbox2d[2])/2, (bbox2d[1]+bbox2d[3])/2
            centers_2d.append([int(center_x), int(center_y)])
        return centers_2d
    
    def generate_depth_weight(self, frame_id, minimum_heat = 0.1):
        centers_2d = self.get_centers_2d(frame_id)
        h, w = self.data[frame_id]["image"]["image_shape"]
        depth_weight = np.zeros([h, w])+minimum_heat
        for x, y in centers_2d:
            depth_weight = draw_gaussian_to_heatmap(depth_weight, [x, y], self.radius)
        save_path = self.data_root / (str(frame_id).zfill(6) + '.png')
        imageio.imwrite(save_path, (depth_weight*65535).astype(np.uint16)) # 'L' 表示灰度图
    
    def generate_all_depth_weight(self):
        for frame_id in tqdm(self.data.keys(), desc="Processing", unit="item"):
            self.generate_depth_weight(frame_id)
    
if __name__ == "__main__":
    dwg = DepthWeightGenerator('../data/kitti/kitti_infos_train.pkl',
                               data_root='../data/kitti/training/depth_weight',
                               config_file = 'cfgs/kitti_models/CaDDN_weight.yaml')
    dwg.generate_all_depth_weight()
    # depth_weight_file = '/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/training/depth_weight/000003.png'
    # depth_weight = io.imread(depth_weight_file).astype(np.float32)
    # depth_weight /= 65535.0
    # print(depth_weight)
    
    