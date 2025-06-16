"""
将kitti的标签转化为ultralytics的训练的格式
"""
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import yaml



def create_kitti_coco(data_root_path):
    """创建kitti_coco数据集

    Args:
        data_root_path (_type_): data数据集
    """
    data_root_path = Path(data_root_path)
    label_folder = data_root_path / "kitti" / "training" / "label_2"
    save_folder = data_root_path / "kitti_coco"
    # 创建图片和标签文件夹
    images_save_folder = save_folder / "images"
    images_save_folder.mkdir(parents=True, exist_ok=True)
    label_save_folder = save_folder / "labels"
    label_save_folder.mkdir(parents=True, exist_ok=True)
    
    # 存储所有的图片训练和验证集
    image_trainval_save_foler = images_save_folder / "trainval"
    image_trainval_save_foler.mkdir(parents=True, exist_ok=True)
    label_trainval_save_folder = label_save_folder / "trainval"
    label_trainval_save_folder.mkdir(parents=True, exist_ok=True)
    
    class_set = dict()
    class_num = 0
    # 读取kitti的label_2文件夹中的所有txt文件
    # 每个txt文件的格式为：type, truncation, occlusion, alpha, x1, y1, x2, y2, h, w, l, tx, ty, tz, ry
    for label_file in tqdm(list(label_folder.glob("*.txt"))):
        image_path = str(label_file).replace("label_2", "image_2").replace(".txt", ".png")
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        coco_labels = []
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    line_split = line.split(" ")
                    class_name = line_split[0]
                    if class_name not in class_set:
                        class_set[class_name] = class_num
                        class_num += 1
                    x1, y1, x2, y2 = line_split[4:8]
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    x_center, y_center = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                    w, h = x2 - x1, y2 - y1
                    w, h = w / width, h / height
                    coco_labels.append([class_set[class_name], x_center, y_center, w, h])
                    # coco_labels.append([0, x_center, y_center, w, h])
                    
        with open(label_trainval_save_folder / f"{label_file.stem}.txt", "w") as f:
            for coco_label in coco_labels:
                f.write(" ".join([str(x) for x in coco_label]) + "\n")
        
        # 将图片软链接
        link_path = image_trainval_save_foler / f"{label_file.stem}.png"
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(image_path, link_path)

    # 生成 YAML 文件
    reverse_class_set = {v: k for k, v in class_set.items()}
    yaml_data = {
        "path": str(save_folder),
        "train": "images/train",
        "val": "images/val",
        "test": "",
        "names": reverse_class_set
    }
    yaml_file_path = save_folder / "kitti_coco.yaml"
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

def split_train_val(data_root_path):
    data_root_path = Path(data_root_path)
    save_folder = data_root_path / "kitti_coco"
    # 创建图片和标签文件夹
    images_save_folder = save_folder / "images"
    images_save_folder.mkdir(parents=True, exist_ok=True)
    label_save_folder = save_folder / "labels"
    label_save_folder.mkdir(parents=True, exist_ok=True)
    
    image_trainval_save_folder = images_save_folder / "trainval"
    label_trainval_save_folder = label_save_folder / "trainval"
    
    # 存储所有的图片训练和验证集
    image_train_save_foler = images_save_folder / "train"
    image_train_save_foler.mkdir(parents=True, exist_ok=True)
    label_train_save_folder = label_save_folder / "train"
    label_train_save_folder.mkdir(parents=True, exist_ok=True)
    
    image_val_save_folder = images_save_folder / "val"
    image_val_save_folder.mkdir(parents=True, exist_ok=True)
    label_val_save_folder = label_save_folder / "val"
    label_val_save_folder.mkdir(parents=True, exist_ok=True)
    
    train_imagesets = data_root_path / "kitti/ImageSets/train.txt"
    val_imagesets = data_root_path / "kitti/ImageSets/val.txt"
    
    with open(train_imagesets, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                if os.path.islink(image_train_save_foler / f"{line}.png") or os.path.exists(image_train_save_foler / f"{line}.png"):
                    os.remove(image_train_save_foler / f"{line}.png")
                if os.path.islink(label_train_save_folder / f"{line}.txt") or os.path.exists(label_train_save_folder / f"{line}.txt"):
                    os.remove(label_train_save_folder / f"{line}.txt")
                os.symlink(image_trainval_save_folder / f"{line}.png", image_train_save_foler / f"{line}.png")
                os.symlink(label_trainval_save_folder / f"{line}.txt", label_train_save_folder / f"{line}.txt")
    with open(val_imagesets, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                if os.path.islink(image_val_save_folder / f"{line}.png") or os.path.exists(image_val_save_folder / f"{line}.png"):
                    os.remove(image_val_save_folder / f"{line}.png")
                if os.path.islink(label_val_save_folder / f"{line}.txt") or os.path.exists(label_val_save_folder / f"{line}.txt"):
                    os.remove(label_val_save_folder / f"{line}.txt")
                os.symlink(image_trainval_save_folder / f"{line}.png", image_val_save_folder / f"{line}.png")
                os.symlink(label_trainval_save_folder / f"{line}.txt", label_val_save_folder / f"{line}.txt")

if __name__ == "__main__":
    create_kitti_coco("/home/zty/Project/DeepLearning/OpenPCDet/data")
    split_train_val("/home/zty/Project/DeepLearning/OpenPCDet/data")