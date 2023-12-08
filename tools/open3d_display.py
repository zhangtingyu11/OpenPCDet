import open3d as o3d
import numpy as np

# lidar_points = np.fromfile('/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/gt_database/006926_Car_0.bin', dtype=np.float32).reshape(-1, 4)
lidar_points = np.fromfile('/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/training/velodyne_reduced/000004.bin', dtype=np.float32).reshape(-1, 4)

added_points = np.fromfile('/home/zty/Project/DeepLearning/OpenPCDet/added_lidar.bin', dtype=np.float32).reshape(-1, 3)
points_3dbox = np.fromfile('/home/zty/Project/DeepLearning/OpenPCDet/box_corners.bin', dtype=np.float32).reshape(8, 3)

lidar_points = lidar_points[:, :3]
vis = o3d.visualization.Visualizer()
vis.create_window()

#! open3d显示lidar
new_points = np.zeros_like(lidar_points)
new_points[:, 0] = -lidar_points[:, 1]
new_points[:, 1] = lidar_points[:, 0]
new_points[:, 2]  = lidar_points[:, 2]
new_point_cloud = o3d.geometry.PointCloud()
new_point_cloud.points = o3d.utility.Vector3dVector(new_points[:, :3])

new_point_cloud.paint_uniform_color([0, 0, 0])

new_points_3dbox = np.zeros_like(points_3dbox)
new_points_3dbox[:, 0] = -points_3dbox[:, 1]
new_points_3dbox[:, 1] = points_3dbox[:, 0]
new_points_3dbox[:, 2] = points_3dbox[:, 2]

lidar_points = lidar_points[:, :3]
lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                        [0, 4], [1, 5], [2, 6], [3, 7]])
#设置点与点之间线段的颜色
colors = np.array([[0, 0, 1] for j in range(len(lines_box))])
#创建Bbox候选框对象
line_set = o3d.geometry.LineSet()
#将八个顶点连接次序的信息转换成o3d可以使用的数据类型
line_set.lines = o3d.utility.Vector2iVector(lines_box)
#设置每条线段的颜色
line_set.colors = o3d.utility.Vector3dVector(colors)
#把八个顶点的空间信息转换成o3d可以使用的数据类型
line_set.points = o3d.utility.Vector3dVector(new_points_3dbox)
#将矩形框加入到窗口中
vis.add_geometry(line_set)
vis.add_geometry(new_point_cloud)
vis.get_render_option().line_width = 9
vis.get_render_option().point_size = 2
vis.run()