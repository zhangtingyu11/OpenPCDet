import pickle
from PIL import Image
import numpy as np
import torch
import imageio
from pathlib import Path

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
    def __init__(self, pickle_file, class_names=['Car', 'Pedestrian', 'Cyclist'], radius = 10, data_root = None):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            self.data = {}
            for item in data:
                frame_id = item["image"]["image_idx"]
                self.data[frame_id] = item
        self.class_names = class_names
        self.radius = radius
        self.data_root = Path(data_root)
    
    def get_centers_2d(self, frame_id):
        centers_2d = []
        frame_id = str(frame_id).zfill(6)
        data = self.data[frame_id]
        annos = data['annos']
        for idx, class_name in enumerate(annos["name"]):
            if class_name not in self.class_names:
                continue
            bbox2d = annos["bbox"][idx]
            center_x, center_y = (bbox2d[0]+bbox2d[2])/2, (bbox2d[1]+bbox2d[3])/2
            centers_2d.append([int(center_x), int(center_y)])
        return centers_2d
    
    def generate_depth_weight(self, frame_id):
        centers_2d = self.get_centers_2d(frame_id)
        h, w = self.data[frame_id]["image"]["image_shape"]
        depth_weight = np.zeros([h, w])
        for x, y in centers_2d:
            depth_weight = draw_gaussian_to_heatmap(depth_weight, [x, y], self.radius)
        save_path = self.data_root / (str(frame_id).zfill(6) + '.png')
        imageio.imwrite(save_path, (depth_weight*65535).astype(np.uint16)) # 'L' 表示灰度图
        
    
    def generate_all_depth_weight(self):
        for frame_id in self.data:
            self.generate_depth_weight(frame_id)
    
if __name__ == "__main__":
    dwg = DepthWeightGenerator('/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/kitti_infos_train.pkl',
                               data_root='/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/training/depth_weight')
    dwg.generate_all_depth_weight()
    
    