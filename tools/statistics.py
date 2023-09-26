import os
from pcdet.utils.object3d_kitti import get_objects_from_label
from pathlib import Path
from collections import *
import pickle
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt

def get_Imagesets(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        lines[idx] = line.strip().zfill(6)
    lines = set(lines)
    return lines

def count_kitti(path, frameset):
    count = {
        'Car': 0,
        'Pedestrian': 0,
        'Cyclist': 0
    }
    for filename in os.listdir(path):
        basename = filename.split('.')[0]
        frame_idx, *class_type, idx = basename.split('_')
        if(frame_idx not in frameset):
            continue
        class_type = "_".join(class_type)
        dataroot = Path("../data/kitti")
        label_txt = dataroot / 'training' / 'label_2' / (frame_idx+'.txt')
        #* 根据标注文件获取GT信息0
        objects = get_objects_from_label(str(label_txt))
        obj = objects[int(idx)]
        occlusion = obj.occlusion
        truncation = obj.truncation
        if occlusion>0 or truncation > 0:
            continue
        if(class_type in count):
            count[class_type]+=1
    return count

def count_nuscenes_lidar(path):
    count = OrderedDict()
    classes = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    for c in classes:
        count[c] = 0
    info_file = "../data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl"
    with open(info_file, 'rb') as f:
        infos = pickle.load(f)
    nusc = NuScenes(version="v1.0-trainval", dataroot="../data/nuscenes/v1.0-trainval", verbose=True)
    for filename in os.listdir(path):
        basename = filename.split('.')[0]
        frame_idx, *class_type, idx = basename.split('_')
        class_type = "_".join(class_type)
        gt_box_token = infos[int(frame_idx)]["gt_boxes_token"][int(idx)]
        #* 当前物体的标注信息
        ann_rec = nusc.get('sample_annotation', gt_box_token)    
        #* 当前物体的可见度
        visibility = ann_rec["visibility_token"]
        if visibility != '4':
            continue
        if(class_type in count):
            count[class_type]+=1
    return count

def count_nuscenes_ours(path):
    count = OrderedDict()
    classes = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    for c in classes:
        count[c] = 0
    info_file = "../data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl"
    with open(info_file, 'rb') as f:
        infos = pickle.load(f)
    nusc = NuScenes(version="v1.0-trainval", dataroot="../data/nuscenes/v1.0-trainval", verbose=True)
    son_paths = ["/../image_gt_database_train_CAM_BACK",
                 "/../image_gt_database_train_CAM_BACK_LEFT",
                 "/../image_gt_database_train_CAM_BACK_RIGHT",
                 "/../image_gt_database_train_CAM_FRONT",
                 "/../image_gt_database_train_CAM_FRONT_LEFT",
                 "/../image_gt_database_train_CAM_FRONT_RIGHT",]
    s = set()
    for son_path in son_paths:
        cur_path = path+son_path
        for filename in os.listdir(cur_path):
            s.add(filename)
    for filename in s:
        basename = filename.split('.')[0]
        frame_idx, *class_type, idx = basename.split('_')
        class_type = "_".join(class_type)
        gt_box_token = infos[int(frame_idx)]["gt_boxes_token"][int(idx)]
        #* 当前物体的标注信息
        ann_rec = nusc.get('sample_annotation', gt_box_token)    
        #* 当前物体的可见度
        visibility = ann_rec["visibility_token"]
        if visibility != '4':
            continue
        if(class_type in count):
            count[class_type]+=1
    return count
        
def plot_kitti():
    frameset = get_Imagesets("../data/kitti/ImageSets/train.txt")
    lidar_count = count_kitti("../data/kitti/gt_database", frameset)
    kins_count = count_kitti("../data/kitti/image_gt_database_trainKINS", frameset)
    ours_count = count_kitti("../data/kitti/image_gt_database_train", frameset)
    # 数据
    categories = ['Car', 'Pedestrian', 'Cyclist']
    data = {
        'lidar': lidar_count.values(),  # 示例数据，请替换为你自己的数据
        'kins': kins_count.values(),
        'ours': ours_count.values()
    }
    colors = ["#63b2ee", "#76da91", "#f8cb7f"]

    # 设置柱状图的宽度
    bar_width = 0.2

    # 生成x轴坐标位置
    x = range(len(categories))

    # 创建子图
    fig, ax = plt.subplots()

    # 为每个数据集绘制柱状图
    for i, (label, values) in enumerate(data.items()):
        x_shifted = [pos + i * bar_width for pos in x]
        ax.bar(x_shifted, values, width=bar_width, label=label, color=colors[i])
        for j, value in enumerate(values):
            ax.text(x_shifted[j], value + 1, str(value), ha='center')

    # 设置x轴标签和标题
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    ax.set_title('Different Sample Strategies(KITTI)')

    # 设置x轴标签
    ax.set_xticks([pos + bar_width for pos in x])
    ax.set_xticklabels(categories)

    # 添加图例
    ax.legend()

    # 显示图形
    plt.tight_layout()
    plt.savefig('Different_Sample_Strategies(nuScenes).eps', format='eps')
    plt.show()

def plot_nuscenes():
    nuscenes_lidar_count = count_nuscenes_lidar("../data/nuscenes/v1.0-trainval/gt_database_10sweeps_withvelo")
    nuscenes_lidar_ours = count_nuscenes_ours("../data/nuscenes/v1.0-trainval/gt_database_10sweeps_withvelo")
    # print(nuscenes_lidar_count)
    # print(nuscenes_lidar_ours)
    

    # 数据
    categories = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    map = {
        'car':"car",
        'truck':'truck',
        'construction_vehicle': "cv",
        'bus':"bus",
        "trailer":"trailer", 
        "barrier": "barrier",
        "motorcycle": "motor",
        "bicycle": "bicycle",
        "pedestrian":"ped",
        "traffic_cone":"tc"
    }
    data = {
        'lidar': nuscenes_lidar_count.values(),  # 示例数据，请替换为你自己的数据
        # 'kins': kins_count.values(),
        'ours': nuscenes_lidar_ours.values()
    }
    colors = ["#63b2ee", "#76da91"]

    # 设置柱状图的宽度
    bar_width = 0.2

    # 生成x轴坐标位置
    x = range(len(categories))

    # 创建子图
    fig, ax = plt.subplots()

    # 为每个数据集绘制柱状图
    for i, (label, values) in enumerate(data.items()):
        # label = map[label]
        x_shifted = [pos + i * bar_width for pos in x]
        ax.bar(x_shifted, values, width=bar_width, label=label, color=colors[i])
        for j, value in enumerate(values):
            ax.text(x_shifted[j], value + 1, str(value), ha='center')

    # 设置x轴标签和标题
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    ax.set_title('Different Sample Strategies(nuScenes)')

    # 设置x轴标签
    ax.set_xticks([pos + bar_width for pos in x])
    ax.set_xticklabels(map.values())

    # 添加图例
    ax.legend()

    # 显示图形
    plt.tight_layout()
    plt.savefig('Different_Sample_Strategies(nuScenes).eps', format='eps')
    plt.show()

        
if __name__ == "__main__":
    plot_kitti()
    plot_nuscenes()

    

    
    
        