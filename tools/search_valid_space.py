import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
from pcdet.utils import calibration_kitti
from pathlib import Path
import yaml

#* 配置中文字体
_cjk_fonts = [f for f in fm.fontManager.ttflist if 'CJK' in f.name]
if _cjk_fonts:
    plt.rcParams['font.sans-serif'] = [_cjk_fonts[0].name] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

class SearchValidSpace:
    def __init__(self, cfg_file) -> None:
        #* 从配置文件中读取数据
        self.cfg = self.cfg_from_yaml_file(cfg_file)
        
        #* 创建画布
        fig_size = self.cfg["FIG_SIZE"]
        self.fig = plt.figure(figsize = fig_size)
        self.ax = self.fig.gca()
        
        #* 画图的中心点, 默认以(0, 0)为中心点
        self.center = self.cfg["CENTER"]
        
        #* 三种区域的颜色        
        self.direct_occupied_color = self.cfg['DIRECT_OCCUPIED_COLOR']
        self.indirect_occupied_color = self.cfg['INDIRECT_OCCUPIED_COLOR']
        self.free_space_color = self.cfg['FREE_SAPCE_COLOR']
        
        #* 数据集根目录
        data_root = self.cfg["DATA_ROOT"] 
        self.data_root = Path(data_root)
        
        self.choosen_frame_id = self.cfg["CHOOSEN_FRAME_ID"]

        self.split = self.cfg["SPLIT"]
        if(self.split in ["train", "val", "trainval"]):
            self.split_dir = "training"

        #* 生成各个文件的路径
        self.bin_file = self.data_root / self.split_dir / 'velodyne' / (self.choosen_frame_id+'.bin')
        self.calib_file = self.data_root / self.split_dir / 'calib' / (self.choosen_frame_id+'.txt')
        self.plane_file = self.data_root / self.split_dir / 'planes' / (self.choosen_frame_id+'.txt')
        
        #* LIDAR线数
        lidar_lines = self.cfg["LIDAR_LINES"]
        #* 每个angle块占据多少角度
        self.angle_interval = 360/lidar_lines
        #* 以竖直向前为0度, 角度范围为-self.angle_limit ~ self.angle_limit
        self.angle_limit = self.cfg["ANGLE_LIMIT"]
        #* 要分多少个角度块
        self.angle_num = int(self.angle_limit*2/self.angle_interval)
        
        #* 距离范围是0~self.range_limit
        self.range_limit = self.cfg["RANGE_LIMIT"]
        #* 要分多少个距离块
        self.range_num = self.cfg["RANGE_NUM"]
        #* 每个range块占据多少距离
        self.range_interval = self.range_limit/self.range_num
        
        self.height_limit = self.cfg["HEIGHT_LIMIT"]

    @staticmethod
    def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.safe_load(f)
        return config
    
    def get_lidar_data(self, filename=None):
        """读取lidar的bin数据

        Args:
            filename (_type_, optional): 如果是None, 就读取self.bin_file. Defaults to None.

        Returns:
            _type_: [N*4]的点云
        """
        bin_file = self.bin_file if filename is None else filename
        bin_file = str(bin_file)
        lidar_points = np.fromfile(bin_file, dtype=np.float32)
        return lidar_points.reshape(-1, 4)    

    def get_calib_data(self, filename=None):
        """获取标定数据

        Args:
            filename (_type_, optional): 标定文件. Defaults to None.

        Returns:
            _type_: 获取到的标定类
        """
        calib_file = self.calib_file if filename is None else filename
        calib_file = str(calib_file)
        calib = calibration_kitti.Calibration(calib_file)
        return calib

    def get_plane_data(self, filename = None):
        plane_file = self.plane_file if filename is None else filename
        plane_file = str(plane_file)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        plane_data = lines[-1].strip().split()
        plane_data = list(map(float, plane_data))
        return plane_data

    def filter_lidar_points(self, points=None, calib=None, road_planes=None):
        lidar_points = self.get_lidar_data() if points is None else points
        lidar_calib = self.get_calib_data() if calib is None else points
        lidar_planes = self.get_plane_data() if road_planes is None else road_planes
        a, b, c, d = lidar_planes
        points_rect = lidar_calib.lidar_to_rect(lidar_points[:, :3])
        #* 求得每个LiDAR点所处的地面高度
        points_rect_height = (-d - a * points_rect[:, 0] - c * points_rect[:, 2]) / b
        #* 必须要高于地面高度才会保留
        valid_mask = points_rect_height > points_rect[:, 1]+self.height_limit
        points_np = lidar_points[valid_mask]
        valid_mask1 = []
        for point in points_np:
            #* 计算距离lidar的距离, 并判断距离索引
            dis = np.sqrt(point[0] * point[0] + point[1] * point[1])
            #* 计算距离索引
            range_idx = int(dis/self.range_interval)
            #* 超过所有就不考虑
            if(range_idx < 0 or range_idx >= self.range_num): 
                valid_mask1.append(False)
                continue
            
            #* 计算角度值，并判断角度索引
            degree = np.arctan2(point[0], -point[1]) * 180 / np.pi
            if(degree < 45):
                valid_mask1.append(False)
                continue  
            #* angle_idx逆时针递增
            angle_idx = int((degree-self.start_angle)/self.angle_interval)
            if(angle_idx < 0 or angle_idx >= self.angle_num): 
                valid_mask1.append(False)
                continue
            valid_mask1.append(True)
        valid_mask1 = np.array(valid_mask1, dtype=np.bool_)
        return points_np[valid_mask1]

    def plot_lidar(self, points):
        self.ax.scatter(-points[:, 1], points[:, 0], s=0.1, c='#000000')

    def show(self):
        plt.xlim(-60, 60)
        plt.ylim(0, 82)
        
        #* angle text
        for idx in range(self.angle_num):
            cur_angle = self.start_angle + idx * self.angle_interval
            text_x = np.cos(np.deg2rad(180-cur_angle)) * (self.range_limit + 3)
            text_y = np.sin(np.deg2rad(180-cur_angle)) * (self.range_limit + 3)
            self.ax.text(text_x, text_y, str(round(180-cur_angle, 2)), fontsize=20, 
                         color='black', ha='center', va='center', rotation=90-cur_angle)
        text_x = np.cos(np.deg2rad(90-self.angle_limit)) * (self.range_limit + 3)
        text_y = np.sin(np.deg2rad(90-self.angle_limit)) * (self.range_limit + 3)
        self.ax.text(text_x, text_y, str(round(90-self.angle_limit, 2)), fontsize=20, 
                     color='black', ha='center', va='center', rotation=-self.angle_limit)
        
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
            self.ax.text(text_x, text_y, str(round(cur_range, 2)), fontsize=20, 
                         color='black', ha='center', va='center')
        
        text_x = np.cos(np.deg2rad(90+self.angle_limit)) * (self.range_limit)-2
        text_y = np.sin(np.deg2rad(90+self.angle_limit)) * (self.range_limit)-1
        x_points.append(text_x+2)
        y_points.append(text_y+1)
        self.ax.text(text_x, text_y, str(round(self.range_limit, 2)), fontsize=20, 
                     color='black', ha='center', va='center')
        
        self.ax.scatter(x_points, y_points)
        # 显示图形
        plt.axis('off')
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles[::-1], labels[::-1], loc = 'lower right', fontsize=28)
        plt.savefig('valid_space.pdf', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
    
    def plot_valid_area(self):
        self.start_angle = 90-self.angle_limit
        points = self.filter_lidar_points()

        #* 起始的角度

        #* occupy数组, 默认都是0表示未占用, 大小为 angle_num * range_num
        occupied = np.zeros((self.angle_num, self.range_num), dtype=np.uint8)
        for point in points:
            #* 计算距离lidar的距离, 并判断距离索引
            dis = np.sqrt(point[0] * point[0] + point[1] * point[1])
            #* 计算距离索引
            range_idx = int(dis/self.range_interval)
            #* 超过所有就不考虑
            if(range_idx < 0 or range_idx >= self.range_num): 
                continue
            
            #* 计算角度值，并判断角度索引
            degree = np.arctan2(point[0], -point[1]) * 180 / np.pi
            #* angle_idx逆时针递增
            angle_idx = int((degree-self.start_angle)/self.angle_interval)
            if(angle_idx < 0 or angle_idx >= self.angle_num): 
                continue
            occupied[angle_idx][range_idx] = 1
        vis = {
            '直接遮挡': False,
            '间接遮挡': False,
            '空闲区域': False
        }
        
        for angle_idx in range(self.angle_num):
            #* 是否出现已经被占用的红色区域
            vis_flag =False
            angle = angle_idx*self.angle_interval + self.start_angle
            for range_idx in range(self.range_num):
                rang = (range_idx+1)*self.range_interval
                if(occupied[angle_idx][range_idx]):
                    wedge = patches.Wedge(self.center, rang, angle, angle+self.angle_interval,
                                        width=self.range_interval, fill=True, color=self.direct_occupied_color, alpha=0.5,
                                        label = '直接遮挡' if not vis['直接遮挡'] else None)
                    #* 这个角度已经出现了被占用的区域
                    vis_flag = True
                    vis['直接遮挡'] = True
                elif(vis_flag):
                    wedge = patches.Wedge(self.center, rang, angle, angle+self.angle_interval,
                                        width=self.range_interval, fill=True, color=self.indirect_occupied_color, alpha=0.5,
                                        label = '间接遮挡' if not vis['间接遮挡'] else None)
                    occupied[angle_idx][range_idx] = 2
                    vis['间接遮挡'] = True
                else:
                    wedge = patches.Wedge(self.center, rang, angle, angle+self.angle_interval,
                                width=self.range_interval, fill=True, color=self.free_space_color, alpha=0.5,
                                label = '空闲区域' if not vis['空闲区域'] else None)
                    vis['空闲区域'] = True
                self.ax.add_patch(wedge)
        self.plot_lidar(points)

if __name__ == '__main__':
    svs = SearchValidSpace("cfgs/dataset_configs/database_generate_kitti.yaml")
    svs.plot_valid_area()
    svs.show()
    