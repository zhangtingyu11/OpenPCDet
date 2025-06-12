import mayavi.mlab as mlab
import numpy as np
import os
import torch
import cv2
import pickle
from visual_utils.visualize_utils import boxes_to_corners_3d
from pcdet.utils.box_utils import boxes3d_kitti_camera_to_lidar
from pcdet.utils import calibration_kitti, common_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from demo import DemoDataset
from pathlib import Path
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
logger = common_utils.create_logger()
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

candidate_class = ["Car", "Pedestrian", "Cyclist"]
class Mayavi_Visualizer:
    def __init__(self, 
                 root_path, 
                 cfg_file, 
                 ckpt_file,
                 velodyne_path,
                 bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), size=(600, 600)):
        self.fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)
        self.root_path = root_path
        cfg_from_yaml_file(cfg_file, cfg)
        self.cfg = cfg
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(velodyne_path), ext=".bin", logger=logger
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        self.model = model
    
    def draw_gt_info(self, index):
        lidar_filename = os.path.join(self.root_path, 'velodyne', '%06d.bin' % index)
        label_filename = os.path.join(self.root_path, 'label_2', '%06d.txt' % index)
        calib_filename = os.path.join(self.root_path, 'calib', '%06d.txt' % index)
        image_filename = os.path.join(self.root_path, 'image_2', '%06d.png' % index)
        img = cv2.imread(image_filename)
        img_shape = img.shape
        self.draw_point_cloud_fromfile(lidar_filename, calib_filename, img_shape)
        self.draw_gt_boxes_fromfile(label_filename, calib_filename)
        
    def draw_pred_info(self, index):
        lidar_filename = os.path.join(self.root_path, 'velodyne', '%06d.bin' % index)
        label_filename = os.path.join(self.root_path, 'label_2', '%06d.txt' % index)
        calib_filename = os.path.join(self.root_path, 'calib', '%06d.txt' % index)
        image_filename = os.path.join(self.root_path, 'image_2', '%06d.png' % index)
        img = cv2.imread(image_filename)
        img_shape = img.shape
        calib = calibration_kitti.Calibration(calib_filename)
        with torch.no_grad():
            data_dict = self.demo_dataset[index]
            pts_rect = calib.lidar_to_rect(data_dict["points"][:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            data_dict["points"] = data_dict["points"][fov_flag]
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            ref_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            ref_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            ref_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            print(ref_labels)
            ref_corners3d = boxes_to_corners_3d(ref_boxes)
            self.draw_corners3d(ref_corners3d, fig=self.fig, color=(0, 0, 1), max_num=100, cls = ref_scores)
            if ref_labels is None:
                self.draw_corners3d(ref_corners3d, fig=self.fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
            else:
                for k in range(ref_labels.min(), ref_labels.max() + 1):
                    cur_color = tuple(box_colormap[k % len(box_colormap)])
                    mask = (ref_labels == k)
                    self.draw_corners3d(ref_corners3d[mask], fig=self.fig, color=cur_color, cls=ref_scores[mask], max_num=100)
        
    def draw_point_cloud_fromfile(self, lidar_filename, calib_filename, img_shape):
        pts = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
        calib = calibration_kitti.Calibration(calib_filename)
        pts_rect = calib.lidar_to_rect(pts[:, 0:3])
        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        pts = pts[fov_flag]
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=self.fig)
        
    def get_fov_flag(self, pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag
    
    def draw_gt_boxes_fromfile(self, label_filename, calib_filename):
        calib = calibration_kitti.Calibration(calib_filename)
        gt_boxes_dict = {}
        with open(label_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_info = line.split()
                gt_class = line_info[0]
                truncated = float(line_info[1])
                occluded = int(line_info[2])
                alpha = float(line_info[3])
                bbox_left, bbox_top, bbox_right, bbox_bottom = [float(x) for x in line_info[4:8]]
                height, width, length = [float(x) for x in line_info[8:11]]
                cam_x, cam_y, cam_z= [float(x) for x in line_info[11:14]]
                lidar_x, lidar_y, lidar_z, length, width, height, heading = \
                    boxes3d_kitti_camera_to_lidar(np.array([[cam_x, cam_y, cam_z, length, height, width, alpha]]), 
                                                  calib)[0]
                rotation_y = float(line_info[14])
                if gt_class not in gt_boxes_dict:
                    gt_boxes_dict[gt_class] = []  
                gt_boxes_dict[gt_class].append(np.array([lidar_x, lidar_y, lidar_z, length, width, height, heading]).reshape(1, -1))

        for gt_class, gt_boxes in gt_boxes_dict.items():
            gt_boxes_dict[gt_class] = np.concatenate(gt_boxes, axis=0)
        
        for gt_class, gt_boxes in gt_boxes_dict.items():
            if gt_class not in candidate_class:
                continue
            corners3d = boxes_to_corners_3d(gt_boxes)
            self.draw_corners3d(corners3d, fig=self.fig, color=(0, 0, 1), max_num=100, cls = [gt_class] * gt_boxes.shape[0])
            
    def draw_corners3d(self, corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
        """
        :param corners3d: (N, 8, 3)
        :param fig:
        :param color:
        :param line_width:
        :param cls:
        :param tag:
        :param max_num:
        :return:
        """
        num = min(max_num, len(corners3d))
        for n in range(num):
            b = corners3d[n]  # (8, 3)

            if cls is not None:
                if isinstance(cls, np.ndarray):
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=self.fig)
                else:
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=self.fig)

            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                            line_width=line_width, figure=self.fig)

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                            line_width=line_width, figure=self.fig)

                i, j = k, k + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                            line_width=line_width, figure=self.fig)

            i, j = 0, 5
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=self.fig)
            i, j = 1, 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=self.fig)

        
if __name__ == "__main__":
    mayavi_vis = Mayavi_Visualizer(
        root_path="/home/grapymage/Projects/OpenPCDet/data/kitti/training",
        cfg_file = "/home/grapymage/Projects/OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_car.yaml",
        ckpt_file = "/home/grapymage/Projects/OpenPCDet/ckpt/voxel_rcnn_car_84.54.pth",
        velodyne_path = "/home/grapymage/Projects/OpenPCDet/data/kitti/training/velodyne"
    )
    index = 6
    mayavi_vis.draw_gt_info(index)
    mayavi_vis.draw_pred_info(index)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    mlab.show()

