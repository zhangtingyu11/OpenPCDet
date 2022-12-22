import pickle

import os
import copy
import numpy as np
from skimage import io
import torch
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        """初始化DataBaseSampler

        Args:
            root_path (_type_): 数据集的根目录
            sampler_cfg (_type_): 采样的配置
            class_names (_type_): 采样的类别
            logger (_type_, optional): 日志
        """
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)

        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            #* self.root_path是pathlib库中的路径，调用resolve函数可以补全成绝对路径
            db_info_path = self.root_path.resolve() / db_info_path
            if not db_info_path.exists():
                assert len(sampler_cfg.DB_INFO_PATH) == 1
                sampler_cfg.DB_INFO_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_INFO_PATH']
                sampler_cfg.DB_DATA_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_DATA_PATH']
                db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH[0]
                sampler_cfg.NUM_POINT_FEATURES = sampler_cfg.BACKUP_DB_INFO['NUM_POINT_FEATURES']

            with open(str(db_info_path), 'rb') as f:
                #* 读取gt数据库里面的数据，每个类别的名称作为key，对应的样本组成的列表作为value
                #* 每个样本包含以下几个内容：
                #* name:类别的名称， 例如'Pedestrian'
                #* path:当前样本所在的点云文件， 例如'gt_database/000000_Pedestrian_0.bin'
                #* image_idx:当前样本的图像的所以， 例如'000000'
                #* gt_idx:当前样本是所在点云的第几个gt样本，例如2
                #* box3d_lidar:lidar坐标系下的3D包围框[x,y,z,dx,dy,dz,heading]，
                #*      例如array([23.79977226, -8.31220436, -0.48439828,  1.09,  0.72 , 1.96 , -3.32079633])
                #* num_points_in_gt:样本包围框内的点数，例如383
                #* difficulty:该包围框属于哪个难度，0为easy，1为moderate，2为hard， -1为三个之外
                #* bbox:2D包围框[xmin, ymin, xmax, ymax]
                #* score:置信度，默认是-1
                infos = pickle.load(f)
                #* self.db_infos存储对应类别的gt样本数据库, key是类别名，value是gt样本组成的列表
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]
        #* sampler_cfg.PREPARE里存放一些预处理操作， 比如过滤点数较少的样本
        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        #* sampler_cfg.SAMPLE_GROUPS存储每个类别要采样多少个样本， 例如['Car:15', 'Pedestrian:15', 'Cyclist:15']
        for x in sampler_cfg.SAMPLE_GROUPS:
            #* class_name是类别名， sample_num是该类别采样的样本数
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            #* self.sample_class_num的key是类别名，value是采样的样本数
            self.sample_class_num[class_name] = sample_num
            #* self.sample_groups的key是类别名，value是一个字典，包含下列三个key
            #*      sample_num:采样的样本数
            #*      pointer: 这一轮采样中已经采样的样本数,第一次设为和数据库长度一样,可以在sample_with_fixed_number中打乱数据库
            #*      indices: 数据库中的样本的索引排列
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        """过滤某些难度的样本

        Args:
            db_infos (_type_): 输入的gt样本数据库
            removed_difficulty (_type_): 需要移除的样本难度的列表， 例如[-1]

        Returns:
            _type_: 新的gt样本数据库
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        """过滤点数少于阈值的样本

        Args:
            db_infos (_type_): gt sample的数据库
            min_gt_points_list (_type_): 存储过滤的列表， 例如['Car:5', 'Pedestrian:5', 'Cyclist:5']

        Returns:
            _type_: 过滤后新的数据库
        """
        for name_num in min_gt_points_list:
            #* name是要过滤的类别， min_num是要过滤的点的阈值
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:类别名
            sample_group:
                sample_num:采样的样本数
                pointer: 这一轮采样中已经采样的样本数,第一次设为和数据库长度一样,可以在sample_with_fixed_number中打乱数据库
                indices: 数据库中的样本的索引排列
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            #* 第一次进入或者每一次取够了样本数的样本后，就打乱顺序供重新选取
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d], 道路的平面方程, ax+by+cz+d=0, [a,b,c,d]
            calib:

        Returns:
            gt_boxes: 修正中心点Z坐标后的gt框
            mv_height: 每个gt框修正的Z坐标的偏差
        """
        a, b, c, d = road_planes
        #* 在相机坐标系下的包围框中心点坐标
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        #* 计算相机中心点(底面中心点)的真实相机高度
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        #* 计算当前lidar坐标系下的中心点高度
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        #* 之前的中心点和现在中心点的偏差值
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        #* 修正gt框的中心点z坐标
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def copy_paste_to_image_kitti(self, data_dict, crop_feat, gt_number, point_idxes=None):
        kitti_img_aug_type = 'by_depth'
        kitti_img_aug_use_type = 'annotation'

        image = data_dict['images']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        corners_lidar = box_utils.boxes_to_corners_3d(boxes3d)
        if 'depth' in kitti_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int)

        if 'reverse' in kitti_img_aug_type:
            paste_order = paste_order[::-1]

        paste_mask = -255 * np.ones(image.shape[:2], dtype=np.int)
        fg_mask = np.zeros(image.shape[:2], dtype=np.int)
        overlap_mask = np.zeros(image.shape[:2], dtype=np.int)
        depth_mask = np.zeros((*image.shape[:2], 2), dtype=np.float)
        points_2d, depth_2d = data_dict['calib'].lidar_to_img(data_dict['points'][:,:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=image.shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=image.shape[0]-1)
        points_2d = points_2d.astype(np.int)
        for _order in paste_order:
            _box2d = boxes2d[_order]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = crop_feat[_order]
            overlap_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] += \
                (paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] > 0).astype(np.int)
            paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = _order

            if 'cover' in kitti_img_aug_use_type:
                # HxWx2 for min and max depth of each box region
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],0] = corners_lidar[_order,:,0].min()
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],1] = corners_lidar[_order,:,0].max()

            # foreground area of original point cloud in image plane
            if _order < gt_number:
                fg_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = 1

        data_dict['images'] = image

        # if not self.joint_sample:
        #     return data_dict

        new_mask = paste_mask[points_2d[:,1], points_2d[:,0]]==(point_idxes+gt_number)
        if False:  # self.keep_raw:
            raw_mask = (point_idxes == -1)
        else:
            raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < gt_number)
            raw_bg = (fg_mask == 0) & (paste_mask < 0)
            raw_mask = raw_fg[points_2d[:,1], points_2d[:,0]] | raw_bg[points_2d[:,1], points_2d[:,0]]
        keep_mask = new_mask | raw_mask
        data_dict['points_2d'] = points_2d

        if 'annotation' in kitti_img_aug_use_type:
            data_dict['points'] = data_dict['points'][keep_mask]
            data_dict['points_2d'] = data_dict['points_2d'][keep_mask]
        elif 'projection' in kitti_img_aug_use_type:
            overlap_mask[overlap_mask>=1] = 1
            data_dict['overlap_mask'] = overlap_mask
            if 'cover' in kitti_img_aug_use_type:
                data_dict['depth_mask'] = depth_mask

        return data_dict

    def collect_image_crops_kitti(self, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        if True:  # self.point_refine:
            # align calibration metrics for points
            points_ract = data_dict['calib'].img_to_rect(points_2d[:,0], points_2d[:,1], depth_2d)
            points_lidar = data_dict['calib'].rect_to_lidar(points_ract)
            obj_points[:, :3] = points_lidar
            # align calibration metrics for boxes
            box3d_raw = sampled_gt_boxes[idx].reshape(1,-1)
            box3d_coords = box_utils.boxes_to_corners_3d(box3d_raw)[0]
            box3d_box, box3d_depth = sampled_calib.lidar_to_img(box3d_coords)
            box3d_coord_rect = data_dict['calib'].img_to_rect(box3d_box[:,0], box3d_box[:,1], box3d_depth)
            box3d_rect = box_utils.corners_rect_to_camera(box3d_coord_rect).reshape(1,-1)
            box3d_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box3d_rect, data_dict['calib'])
            box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_rect, data_dict['calib'],
                                                                    data_dict['images'].shape[:2])
            sampled_gt_boxes[idx] = box3d_lidar[0]
            sampled_gt_boxes2d[idx] = box2d[0]

        obj_idx = idx * np.ones(len(obj_points), dtype=np.int)

        # copy crops from images
        img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        raw_image = io.imread(img_path)
        raw_image = raw_image.astype(np.float32)
        raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_boxes2d[idx].astype(np.int)
        new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int)
        raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
            new_center = new_box.reshape(2,2).mean(0)
            new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
            new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int)

        img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255

        return new_box, img_crop2d, obj_points, obj_idx

    def sample_gt_boxes_2d_kitti(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None
        # filter out box2d iou > thres
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )

        # sampled_boxes2d = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)
        boxes3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(sampled_boxes, data_dict['calib'])
        sampled_boxes2d = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, data_dict['calib'],
                                                                        data_dict['images'].shape[:2])
        sampled_boxes2d = torch.Tensor(sampled_boxes2d)
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        iou2d1 = box_utils.pairwise_iou(sampled_boxes2d, existed_boxes2d).cpu().numpy()
        iou2d2 = box_utils.pairwise_iou(sampled_boxes2d, sampled_boxes2d).cpu().numpy()
        iou2d2[range(sampled_boxes2d.shape[0]), range(sampled_boxes2d.shape[0])] = 0
        iou2d1 = iou2d1 if iou2d1.shape[1] > 0 else iou2d2

        ret_valid_mask = ((iou2d1.max(axis=1)<self.img_aug_iou_thresh) &
                         (iou2d2.max(axis=1)<self.img_aug_iou_thresh) &
                         (valid_mask))

        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        if mv_height is not None:
            mv_height = mv_height[ret_valid_mask]
        return sampled_boxes2d, mv_height, ret_valid_mask

    def sample_gt_boxes_2d(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None

        if self.img_aug_type == 'kitti':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_kitti(data_dict, sampled_boxes, valid_mask)
        else:
            raise NotImplementedError

        return sampled_boxes2d, mv_height, ret_valid_mask

    def initilize_image_aug_dict(self, data_dict, gt_boxes_mask):
        img_aug_gt_dict = None
        if self.img_aug_type is None:
            pass
        elif self.img_aug_type == 'kitti':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int)
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int)
            gt_crops2d = [data_dict['images'][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        else:
            raise NotImplementedError

        return img_aug_gt_dict

    def collect_image_crops(self, img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        if self.img_aug_type == 'kitti':
            new_box, img_crop2d, obj_points, obj_idx = self.collect_image_crops_kitti(info, data_dict,
                                                    obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx)
            img_aug_gt_dict['crop_boxes2d'].append(new_box)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
            img_aug_gt_dict['obj_index_list'].append(obj_idx)
        else:
            raise NotImplementedError

        return img_aug_gt_dict, obj_points

    def copy_paste_to_image(self, img_aug_gt_dict, data_dict, points):
        if self.img_aug_type == 'kitti':
            obj_points_idx = np.concatenate(img_aug_gt_dict['obj_index_list'], axis=0)
            point_idxes = -1 * np.ones(len(points), dtype=np.int)
            point_idxes[:obj_points_idx.shape[0]] = obj_points_idx

            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_kitti(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'], point_idxes)
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
        else:
            raise NotImplementedError
        return data_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        """将采样的gt框加进场景中

        Args:
            data_dict (_type_): 当前场景的数据信息
                frame_id:帧id
                calib:标定类
                gt_names:gt框的类别名
                gt_boxes:gt包围框信息
                road_plane:道路的平面方程, ax+by+cz+d=0, [a,b,c,d]
                points:点云
                gt_boxes_mask:gt框的掩码, 在识别类别中则为True
            sampled_gt_boxes (_type_): 采样的gt框数据
            total_valid_sampled_dict (_type_): 采样的gt框的全部数据组成列表
            mv_height (_type_, optional): _description_. Defaults to None.
            sampled_gt_boxes2d (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        # convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)

        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)

            assert obj_points.shape[0] == info['num_points_in_gt']
            #* obj_points原本存放的是以obj中心点为原点的点的坐标
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                #* 认定同一包围框内的点的高度偏移量是一致的
                obj_points[:, 2] -= mv_height[idx]

            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)
        #* 将添加的gt框的点拼接起来
        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        #* 移除点云内的包围框的点
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.img_aug_type is not None:
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)
        """
        data_dict的返回值如下:
            frame_id:帧id
            gt_names:gt框的类别名
            gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
            points:点云
            gt_boxes_mask:gt框的掩码, 在识别类别中则为True, 初始的gt_boxes_mask, 这里应该没啥用了
        """
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        sampled_mv_height = []
        sampled_gt_boxes2d = []

        #* self.sample_groups的key是类别名，value是一个字典，字典包含下列三个key
        #*      sample_num:采样的样本数
        #*      pointer: 这一轮采样中已经采样的样本数,第一次设为和数据库长度一样,可以在sample_with_fixed_number中打乱数据库
        #*      indices: 数据库中的样本的索引排列
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                #* sampled_dict是采样的gt样本组成的列表
                #* name:类别的名称， 例如'Pedestrian'
                #* path:当前样本所在的点云文件， 例如'gt_database/000000_Pedestrian_0.bin'
                #* image_idx:当前样本的图像的所以， 例如'000000'
                #* gt_idx:当前样本是所在点云的第几个gt样本，例如2
                #* box3d_lidar:lidar坐标系下的3D包围框[x,y,z,dx,dy,dz,heading]，
                #*      例如array([23.79977226, -8.31220436, -0.48439828,  1.09,  0.72 , 1.96 , -3.32079633])
                #* num_points_in_gt:样本包围框内的点数，例如383
                #* difficulty:该包围框属于哪个难度，0为easy，1为moderate，2为hard， -1为三个之外
                #* bbox:2D包围框[xmin, ymin, xmax, ymax]
                #* score:置信度，默认是-1
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                #* 获取3D框np数组
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'
                
                #* iou1是采样的包围框和原有的包围框的iou
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                #* iou2是采样的包围框和采样的包围框的iou
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                #* 将对角线的位置都变为0
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                #* 可能会出现原有的gt框个数为0的情况
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                #* 求出有效的添加的包围框的mask
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                if self.img_aug_type is not None:
                    sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, valid_mask)
                    sampled_gt_boxes2d.append(sampled_boxes2d)
                    if mv_height is not None:
                        sampled_mv_height.append(mv_height)
                
                valid_mask = valid_mask.nonzero()[0]
                #* 有效的采样的gt框的标注信息
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                #* 有效的采样的gt框
                valid_sampled_boxes = sampled_boxes[valid_mask]

                #* 拓展现有的gt框
                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        if total_valid_sampled_dict.__len__() > 0:
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )

        data_dict.pop('gt_boxes_mask')
        """
        data_dict的返回值如下:
            frame_id:帧id
            gt_names:gt框的类别名
            gt_boxes:gt框, [x,y,z,dx,dy,dz,heading]
            points:点云
        """
        return data_dict
