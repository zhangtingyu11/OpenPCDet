import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import open3d as o3d
import pickle
from pcdet.utils.box_utils import boxes3d_kitti_camera_to_lidar, boxes3d_kitti_camera_to_imageboxes, boxes3d_lidar_to_kitti_camera
from pcdet.utils import calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import cv2

RANGE_LIMIT = 80
HEIGHT_LIMIT = -1.5
ANGLE_LIMIT = 40.69
RANGE_NUM = 100
GT_SAMPLING_NUM=1
FRAME_ID = '007409'
DATA_ROOT = '../data/kitti/training'

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str


class SearchValidSpace:
    def __init__(self, data_root, frame_id, range_limit, height_limit, angle_limit, range_num) -> None:
        #* 创建画布
        self.fig = plt.figure(figsize = (20, 20))
        self.ax = self.fig.gca()
        
        #* 画图的中心点, 默认以(0, 0)为中心点
        self.center = (0, 0)
        
        #* 三种区域的颜色        
        self.DIRECT_OCCUPIED = 'r'
        self.INDIRECT_OCCUPIED = 'b'
        self.FREE_SAPCE = 'g'
        
        self.data_root = data_root 
        self.frame_idx = frame_id
        
        #* 生成各个文件的路径
        self.bin_file = '/'.join([self.data_root, 'velodyne_reduced', frame_id+'.bin'])
        self.calib_file = '/'.join([self.data_root, 'calib', frame_id+'.txt'])
        self.database_file = '/'.join([self.data_root, '..', 'image_database_train.pkl'])
        self.label_file = '/'.join([self.data_root, 'label_2', frame_id+'.txt'])
        self.image_file = '/'.join([self.data_root, 'image_2', frame_id+'.png'])
        self.plane_file = '/'.join([self.data_root, 'planes', frame_id+'.txt'])
        
        self.range_limit = range_limit
        self.height_limit = height_limit
        self.angle_limit = angle_limit
        self.range_num = range_num
    
    def get_lidar_data(self, filename=None):
        """读取lidar的bin数据

        Args:
            filename (_type_, optional): 如果是None, 就读取self.bin_file. Defaults to None.

        Returns:
            _type_: [N*4]的点云
        """
        bin_file = self.bin_file if filename is None else filename
        lidar_points = np.fromfile(bin_file, dtype=np.float32)
        return lidar_points.reshape(-1, 4)    
    
    def get_database_data(self, filename=None):
        """读取数据库数据
        数据库是一个列表, 存了处于不同angle范围的物体, 索引是逆时针递增。

        Returns:
            _type_: _description_
        """
        database_file = filename if filename is not None else self.database_file
        with open(database_file, 'rb') as f:
            database = pickle.load(f)
        return database
    
    def get_calib_data(self, filename=None):
        """获取标定数据

        Args:
            filename (_type_, optional): 标定文件. Defaults to None.

        Returns:
            _type_: 获取到的标定类
        """
        calib_file = self.calib_file if filename is None else filename
        calib = calibration_kitti.Calibration(calib_file)
        return calib
    
    def get_label_data(self, filename = None):
        label_file = self.label_file if filename is None else filename
        with open(label_file, 'r') as f:
            lines = f.readlines()
            objects = [Object3d(line) for line in lines]
            return objects
        
    def get_image_data(self, filename = None):
        image_file = self.image_file if filename is None else filename
        return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    
    def get_plane_data(self, filename = None):
        plane_file = self.plane_file if filename is None else filename
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        plane_data = lines[-1].strip().split()
        plane_data = list(map(float, plane_data))
        return plane_data

    def filter_lidar_points(self, points=None):
        lidar_points = self.get_lidar_data() if points is None else points
        #* 将numpy数组转成open3d格式的点云
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        vis = o3d.visualization.Visualizer()
        #设置窗口标题
        vis.create_window(window_name="kitti")
        #设置点云大小
        vis.get_render_option().point_size = 1
        #设置颜色背景为黑色
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1,1,1])

        #创建点云对象
        #将点云数据转换为Open3d可以直接使用的数据类型
        #设置点的颜色为白色
        point_cloud.paint_uniform_color([0, 0, 0])
        #将点云加入到窗口中
        vis.add_geometry(point_cloud)
        #* 对点云地面进行过滤
        _, inliers = point_cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)

        #* 选择非地面点
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)


        vis.run()
        vis.destroy_window()
        
        #* 转化成numpy数组
        points_np = np.asarray(outlier_cloud.points)
        #* 对点云的高度进行过滤，过滤低于一定高度的点云
        points_np = points_np[points_np[:, 2] > HEIGHT_LIMIT]
        #* 对超过一定距离的点云进行过滤
        points_np = points_np[np.linalg.norm(points_np[:, :2], axis=-1) <= RANGE_LIMIT]
        return points_np

    def plot_lidar(self, points):
        self.ax.scatter(-points[:, 1], points[:, 0], s=0.1, c='#000000')

    def show(self):
        plt.xlim(-53, 53)
        plt.ylim(0, 82)
        start_angle = 90-self.angle_limit
        angle_intervel = (self.angle_limit*2)/self.angle_num
        
        #* angle text
        for idx in range(self.angle_num):
            cur_angle = start_angle + idx * angle_intervel
            text_x = np.cos(np.deg2rad(180-cur_angle)) * (self.range_limit + 3)
            text_y = np.sin(np.deg2rad(180-cur_angle)) * (self.range_limit + 3)
            self.ax.text(text_x, text_y, str(round(180-cur_angle, 2)), fontsize=16, color='black', ha='center', va='center', rotation=90-cur_angle)
        text_x = np.cos(np.deg2rad(90-self.angle_limit)) * (self.range_limit + 3)
        text_y = np.sin(np.deg2rad(90-self.angle_limit)) * (self.range_limit + 3)
        self.ax.text(text_x, text_y, str(round(90-self.angle_limit, 2)), fontsize=16, color='black', ha='center', va='center', rotation=-self.angle_limit)
        
        #* range_text
        x_points, y_points = [], []
        range_split = 5
        range_interval = self.range_limit/range_split
        for idx in range(range_split):
            cur_range = idx*range_interval
            text_x = np.cos(np.deg2rad(90+self.angle_limit)) * (cur_range)-2
            text_y = np.sin(np.deg2rad(90+self.angle_limit)) * (cur_range)-1
            x_points.append(text_x+2)
            y_points.append(text_y+1)
            self.ax.text(text_x, text_y, str(round(cur_range, 2)), fontsize=16, color='black', ha='center', va='center')
        
        text_x = np.cos(np.deg2rad(90+self.angle_limit)) * (self.range_limit)-2
        text_y = np.sin(np.deg2rad(90+self.angle_limit)) * (self.range_limit)-1
        x_points.append(text_x+2)
        y_points.append(text_y+1)
        self.ax.text(text_x, text_y, str(round(self.range_limit, 2)), fontsize=16, color='black', ha='center', va='center')
        
        self.ax.scatter(x_points, y_points)
        # 显示图形
        plt.axis('off')
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles[::-1], labels[::-1], loc = 'lower right', fontsize=16)
        plt.savefig('augmented_lidar.png', bbox_inches='tight', dpi = self.fig.dpi, pad_inches=0.0)
        plt.show()
    
    def plot_valid_area(self):
        points = self.filter_lidar_points()
        angle_limit = self.angle_limit
        #* 64线激光雷达, 计算这个范围覆盖了多少angle
        angle_num = int((angle_limit*2)/(360/64))
        self.angle_num = angle_num
        
        #* 距离分多少个
        range_num = self.range_num

        #* 起始的角度和终止的角度
        start_angle, end_angle = 90-angle_limit, 90+angle_limit

        #* 计算每个角度区间大小
        angle_interval = (end_angle-start_angle)/angle_num
        
        #* 计算每个距离区间的大小
        range_interval = self.range_limit/range_num
        
        #* occupy数组, 默认都是0表示未占用, 大小为 angle_num * range_num
        occupied = np.zeros((angle_num, range_num), dtype=np.bool_)
        for point in points:
            #* 计算距离lidar的距离, 并判断距离索引
            dis = np.sqrt(point[0] * point[0] + point[1] * point[1])
            dis_idx = int(dis/range_interval)
            if(dis_idx<0 or dis_idx>=range_num): continue
            
            #* 计算角度值，并判断角度索引
            degree = np.arctan2(point[0], -point[1]) * 180 / np.pi
            #* angle_idx逆时针递增
            angle_idx = int((degree-start_angle)/angle_interval)
            if(angle_idx<0 or angle_idx>=angle_num): continue
            
            occupied[angle_idx][dis_idx] = True
        
        #* 判断这个角度索引有没有被占用过
        angle_vis = set()
        #* 为了添加标签, 需要设置第一次访问
        occupy_vis = [False] * 3
        
        #* 遍历所有的角度和距离
        r_idx = 0
        while r_idx < range_num:
            angle_idx = 0      
            r = (r_idx+1)*range_interval
            while angle_idx < angle_num:
                angle = angle_idx*angle_interval + start_angle
                if(occupied[angle_idx][r_idx]):
                    wedge = patches.Wedge(self.center, r, angle, angle+angle_interval,
                                        width=range_interval, fill=True, color=self.DIRECT_OCCUPIED, alpha=0.5,
                                        label = 'direct occupation' if not occupy_vis[0] else None)
                    occupy_vis[0] = True
                    angle_vis.add(angle_idx)
                else:
                    if(angle_idx not in angle_vis):
                        wedge = patches.Wedge(self.center, r, angle, angle+angle_interval,
                                    width=range_interval, fill=True, color=self.FREE_SAPCE, alpha=0.5, 
                                    label = 'free space' if not occupy_vis[1] else None)
                        occupy_vis[1] = True
                    else:
                        wedge = patches.Wedge(self.center, r, angle, angle+angle_interval,
                                            width=range_interval, fill=True, color=self.INDIRECT_OCCUPIED, alpha=0.5, 
                                            label = 'indirect occupation' if not occupy_vis[2] else None)  
                        occupy_vis[2] = True
                        occupied[angle_idx][r_idx] = True
                self.ax.add_patch(wedge)
                angle_idx+=1
            r_idx+=1
            
        self.augment_display(points, occupied, angle_interval, start_angle, range_interval)
    
    def augment_display(self, points, occupied, angle_interval, start_angle, range_interval):
        #* 计算非零的索引
        unoccupied_indexs = (occupied==False).nonzero()
        #* -> N * 2
        unoccupied_indexs = np.concatenate([np.expand_dims(unoccupied_indexs[0], axis=-1), np.expand_dims(unoccupied_indexs[1], axis=-1)], axis=-1)
        
        #* 设置采样的概率, 越靠后的位置采样概率越大, 并且最小的10个距离索引不会采样
        decay_factor = 0.05
        probabilities = np.array([np.exp(decay_factor*unoccupied_indexs[i][1]) if unoccupied_indexs[i][1] > 20 else 0 for i in range(unoccupied_indexs.shape[0])])
        probabilities/=probabilities.sum()
        
        #* 采样可以放置的位置
        indexs = np.random.choice(range(unoccupied_indexs.shape[0]), size=GT_SAMPLING_NUM, replace=False, p=probabilities)
        choosen_indexs = unoccupied_indexs[indexs]
        
        #* 用红点画出采样的点
        point_vis = False
        for angle_idx, range_idx in choosen_indexs:
            
            angle = (angle_idx+0.5)*angle_interval+start_angle
            r = (range_idx+0.5)*range_interval
            
            x = np.cos(np.deg2rad(angle)) * r
            y = np.sin(np.deg2rad(angle)) * r
            self.ax.scatter(x, y, c='r', s=5,
                            label = 'choosen position' if not point_vis else None)
            point_vis = True
        
        objects = self.get_label_data()
        boxes_camera_3d_label_array = np.array([[*(obj.loc), obj.l, obj.h, obj.w, obj.ry] for obj in objects])
        label_boxes_2d = []
        for obj in objects:
            if(obj.cls_type=='Car'):
                label_boxes_2d.append(obj.box2d)
            
        origin_calib = self.get_calib_data()
        boxes_lidar_3d_label_array = boxes3d_kitti_camera_to_lidar(boxes_camera_3d_label_array, origin_calib)
        origin_label_box_num = boxes_lidar_3d_label_array.shape[0]
        
        #* 对选取的索引排序，越远的越先采样
        choosen_indexs = sorted(choosen_indexs, key=lambda x: x[1], reverse=True)
        database = self.get_database_data()
        
        origin_image = self.get_image_data()
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2BGRA)
        
        added_boxes_coor = []
        for angle_idx, range_idx in choosen_indexs:
            #* 从这个角度区间随机选一个
            r_idx = int(range_idx//12.5)
            if(len(database[angle_idx][r_idx]) == 0):
                continue
            random_idx = random.randint(0, len(database[angle_idx][r_idx])-1)
            cur_filename = database[angle_idx][r_idx][random_idx]
            
            pre, _ = cur_filename.split('.')
            frame_id, cls_type, idx = pre.split('_')
            
            #* 读取这个样本的标签
            label_file = '../data/kitti/training/label_2/' + frame_id + '.txt'
            added_labels = self.get_label_data(label_file)
            
            #* 读取这个样本的标定文件
            calib_file = '../data/kitti/training/calib/' + frame_id + '.txt'
            plane_file = '../data/kitti/training/planes/' + frame_id + '.txt'
            calib = self.get_calib_data(calib_file)
            
            added_object = added_labels[int(idx)]
            added_object = np.array([*(added_object.loc), added_object.l, added_object.h, added_object.w, added_object.ry])
            added_object = np.expand_dims(added_object, axis=0)
            #* 转到lidar坐标系
            added_object = boxes3d_kitti_camera_to_lidar(added_object, calib)
            #TODO 
            a, b, c, d = self.get_plane_data(plane_file)
            center_cam = calib.lidar_to_rect(added_object[:, 0:3])
            cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
            center_cam[:, 1] = cur_height_cam
            cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
            mv_height = added_object[:, 2] - added_object[:, 5] / 2 - cur_lidar_height
            added_object[:, 2] -= mv_height  # lidar view
            
            
            #* 将其平移到这个扇形的中心
            angle = (angle_idx+0.5)*angle_interval+start_angle
            r = (range_idx+0.5)*range_interval
            center_x = np.cos(np.deg2rad(angle)) * r
            center_y = np.sin(np.deg2rad(angle)) * r
            
            #* 将点云增加到特定位置
            added_bin_file = '../data/kitti/gt_database/'  + pre + '.bin'
            added_points = self.get_lidar_data(added_bin_file)
            added_points[:, 0] += center_y
            added_points[:, 1] += -center_x
            added_points[:, 2] += added_object[0][2]
            
            #* 将障碍物的中心点改变
            added_object[0][0] = center_y
            added_object[0][1] = -center_x
            
            #* 和已经添加的框计算iou, 必须无碰撞
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(added_object[:, 0:7], boxes_lidar_3d_label_array[:, 0:7])
            if(iou.max() == 0):
                #* 添加包围框
                boxes_lidar_3d_label_array = np.concatenate([boxes_lidar_3d_label_array, added_object], axis=0)
                #* 添加对应的点云
                points = np.concatenate([added_points[:, :3], points[:, :3]], axis=0)
                #* 添加对应的图像
                box_camera = boxes3d_lidar_to_kitti_camera(added_object, calib)
                boxes_image = boxes3d_kitti_camera_to_imageboxes(box_camera, calib)[0].astype(np.int32)
                added_image = self.get_image_data('/'.join([self.data_root, '..', 'image_gt_database_train', cur_filename]))
                added_image = cv2.resize(added_image, (boxes_image[2]-boxes_image[0], boxes_image[3]-boxes_image[1]), interpolation=cv2.INTER_LINEAR)
                left, top, right, bottom = boxes_image
                new_left, new_top, new_right, new_bottom = max(0, left), max(0, top), min(right, origin_image.shape[1]), min(bottom, origin_image.shape[0])
                croped_image = origin_image[new_top:new_bottom, new_left:new_right]
                alpha1 = added_image[(new_top-top):(new_bottom-top), (new_left-left):(new_right-left)][:, :, 3]
                mask = alpha1 > 0
                croped_image[mask] = added_image[(new_top-top):(new_bottom-top), (new_left-left):(new_right-left)][mask]
                origin_image[new_top:new_bottom, new_left:new_right] = croped_image
                added_boxes_coor.append([new_top, new_left, new_bottom, new_right])
        
        for new_top, new_left, new_bottom, new_right in added_boxes_coor:
            cv2.rectangle(origin_image, (new_left, new_top), (new_right, new_bottom), color = (0, 0, 255))
        for new_top, new_left, new_bottom, new_right in label_boxes_2d:
            cv2.rectangle(origin_image, (int(new_top), int(new_left)), (int(new_bottom), int(new_right)), color = (0, 255, 0))
        
        cv2.imwrite('augmented_image.png', origin_image[:, :, :3])
        self.plot_lidar(points)
        self.add_boxes(boxes_lidar_3d_label_array, origin_label_box_num)
        
    def add_boxes(self, boxes, label_box_num):
        vis_label = False
        vis_added = False
        for idx, (x, y, z, length, width, height, heading) in enumerate(boxes):
            #* 这是原来的包围框
            if(idx < label_box_num):
                rectangle = patches.Rectangle((-y-width/2,x-length/2), width, length, angle = np.rad2deg(heading), color = '#8F3C2E', alpha=0.7,
                                              rotation_point = 'center',
                                              label = 'original GT' if not vis_label else None)
                vis_label = True
            #* 这是添加的包围框
            else:
                # print([x, y, z, length, width, height, heading], sep= ', ')
                rectangle = patches.Rectangle((-y-width/2,x-length/2), width, length, angle = np.rad2deg(heading), color = '#137BDB', alpha=0.7,
                                              rotation_point = 'center',
                                              label = 'added GT' if not vis_added else None)
                vis_added = True
            self.ax.add_patch(rectangle)

if __name__ == '__main__':
    svs = SearchValidSpace(DATA_ROOT, FRAME_ID, RANGE_LIMIT, HEIGHT_LIMIT, ANGLE_LIMIT, RANGE_NUM)
    svs.plot_valid_area()
    svs.show()

