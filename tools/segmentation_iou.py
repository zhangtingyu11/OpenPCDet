import os
import cv2
from math import *
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

if __name__ == "__main__":
    ours_path = "/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/image_gt_database_train"
    kins_path = "/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/image_gt_database_trainKINS"
    lidar_path = "/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/gt_database"
    lidar_filenames = set(filename.split('.')[0] for filename in os.listdir(lidar_path))
    ours_filenames = set(filename.split('.')[0] for filename in os.listdir(ours_path))
    kins_filenames = set(filename.split('.')[0] for filename in os.listdir(kins_path))
    car_ans = []
    ped_ans = []
    cyc_ans = []
    for filename in tqdm(ours_filenames):
        if(filename not in kins_filenames):
            continue
        ours_file = '/'.join([ours_path, filename+'.png'])
        kins_file = '/'.join([kins_path, filename+'.png'])
        ours_img = cv2.imread(ours_file, cv2.IMREAD_UNCHANGED)
        kins_img = cv2.imread(kins_file, cv2.IMREAD_UNCHANGED)
        ours_idxs = (ours_img[:, :, 3]>0)
        kins_idxs = (kins_img[:, :, 3]>0)
        area_ours = ours_idxs.sum()
        area_kins = kins_idxs.sum()
        inter = (ours_idxs & kins_idxs).sum()
        ua = area_ours+area_kins-inter
        _, t, _ = filename.split('_')
        if t == 'Car':
            car_ans.append(inter/ua)
        elif t == 'Pedestrian':
            ped_ans.append(inter/ua)
        elif t == 'Cyclist':
            cyc_ans.append(inter/ua)
    print(max(car_ans))
    print(min(car_ans))
    print(sum(car_ans)/len(car_ans))
    print(len(car_ans))
    
    print(max(ped_ans))
    print(min(ped_ans))
    print(sum(ped_ans)/len(ped_ans))
    print(len(ped_ans))
    
    print(max(cyc_ans))
    print(min(cyc_ans))
    print(sum(cyc_ans)/len(cyc_ans))
    print(len(cyc_ans))
    total = car_ans+ped_ans+cyc_ans
    print(max(total))
    print(min(total))
    print(sum(total)/len(total))
    print(len(total))

        
        
