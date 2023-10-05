import os
from decimal import Decimal
import random

def influence_of_detector2d():
    scores = [[0, 0.2], [0.6, 1.0], [0.9, 1.0]]
    propotions = [0.5, 1]
    for propotion in propotions:
        for low_score, high_score in scores:
            yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml"
            propotion_line_idx = 69
            low_score_line_idx = 70
            high_score_line_idx = 71
            with open(yaml_file, 'r') as f:
                lines = f.readlines()
                propotion_line = lines[propotion_line_idx]
                splits = propotion_line.split(': ')
                splits[-1] = str(propotion)+'\n'
                lines[propotion_line_idx] = ': '.join(splits)
                
                low_score_line = lines[low_score_line_idx]
                splits = low_score_line.split(': ')
                splits[-1] = str(low_score)+'\n'
                lines[low_score_line_idx] = ': '.join(splits)
                
                high_score_line = lines[high_score_line_idx]
                splits = high_score_line.split(': ')
                splits[-1] = str(high_score)+'\n'
                lines[high_score_line_idx] = ': '.join(splits)
            with open(yaml_file, 'w') as f:
                for line in lines:
                    f.write(line)
            os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml --fix_random_seed 666 --extra_tag retinanet_epoch10_seed666_low_score' + str(low_score) +\
                '_high_score_' + str(high_score) + '_propotion_' + str(propotion) + ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')   
            

def compare():
    config_files = ['/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/retinanet_r18_fpn_1x_coco/20230927_150345/vis_data/config.py', 
                    '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/retinanet_r50_fpn_1x_coco/20230927_155447/vis_data/config.py',
                    '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/cascade-rcnn_r50_fpn_1x_coco/20230928_003043/vis_data/config.py',
                    '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/cascade-rcnn_r50_fpn_20e_coco/20230928_012105/vis_data/config.py']
    weight_files = ['/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/retinanet_r18_fpn_1x_coco/epoch_9.pth', 
                    '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/retinanet_r50_fpn_1x_coco/epoch_12.pth',
                    '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/cascade-rcnn_r50_fpn_1x_coco/epoch_5.pth',
                    '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/cascade-rcnn_r50_fpn_20e_coco/epoch_7.pth'
                    ]
    for i in range(len(config_files)):
        config_file = config_files[i]
        weight_file = weight_files[i]
        tag = config_file.split('/')[8]
        yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra.yaml"
        config_file_line = 64
        weight_file_line = 65
        with open(yaml_file, 'r') as f:
            lines = f.readlines()
            config_line = lines[config_file_line]
            splits = config_line.split(': ')
            splits[-1] = config_file+'\n'
            lines[config_file_line] = ': '.join(splits)
            
            weight_line = lines[weight_file_line]
            splits = weight_line.split(': ')
            splits[-1] = weight_file+'\n'
            lines[weight_file_line] = ': '.join(splits)
        with open(yaml_file, 'w') as f:
            for line in lines:
                f.write(line)
        os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_contra.yaml --fix_random_seed 666 --extra_tag ' + tag + '_epoch10_no_gt_sampling_ours_filter_no_pad_seed666' +\
            ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')   
        
        yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs.yaml"
        config_file_line = 64
        weight_file_line = 65
        with open(yaml_file, 'r') as f:
            lines = f.readlines()
            config_line = lines[config_file_line]
            splits = config_line.split(': ')
            splits[-1] = config_file+'\n'
            lines[config_file_line] = ': '.join(splits)
            
            weight_line = lines[weight_file_line]
            splits = weight_line.split(': ')
            splits[-1] = weight_file+'\n'
            lines[weight_file_line] = ': '.join(splits)
        with open(yaml_file, 'w') as f:
            for line in lines:
                f.write(line)
        os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs.yaml --fix_random_seed 666 --extra_tag ' + tag + '_epoch10_no_gt_sampling_ours_filter_no_pad_seed666' +\
            ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')  

def choose_contra_weight():
    weights = ['0.001', '0.01', '0.1', '1.0']
    
    for weight in weights:
        config_file = '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/retinanet_r18_fpn_1x_coco/20230927_150345/vis_data/config.py'
        weight_file = '/home/zty/Project/DeepLearning/OpenPCDet/mmdetection/work_dirs_single_class/retinanet_r18_fpn_1x_coco/epoch_9.pth'
        tag = config_file.split('/')[8]
        yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra.yaml"
        config_file_line = 64
        weight_file_line = 65
        contra_weight_line = 141
        
        with open(yaml_file, 'r') as f:
            lines = f.readlines()
            config_line = lines[config_file_line]
            splits = config_line.split(': ')
            splits[-1] = config_file+'\n'
            lines[config_file_line] = ': '.join(splits)
            
            weight_line = lines[weight_file_line]
            splits = weight_line.split(': ')
            splits[-1] = weight_file+'\n'
            lines[weight_file_line] = ': '.join(splits)
            
            contra_weight = lines[contra_weight_line]
            splits = contra_weight.split(': ')
            splits[-1] = weight+',\n'
            lines[contra_weight_line] = ': '.join(splits)
        with open(yaml_file, 'w') as f:
            for line in lines:
                f.write(line)
        os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_contra.yaml --fix_random_seed 666 --extra_tag ' + tag + '_epoch1_no_gt_sampling_ours_filter_no_pad_seed666_weight' + weight +\
            ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')   

if __name__=="__main__":
    influence_of_detector2d()
