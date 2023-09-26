import os
from decimal import Decimal
import random

# yaml_file = "/home/public/zty/Project/DeepLearningProject/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra.yaml"
# car_line = 120

# target = Decimal('1')
# start = Decimal('0.001')
# increase = Decimal('10')
# while start <= target:
#     with open(yaml_file, 'r') as f:
#         lines = f.readlines()
#         pre, suff = lines[car_line].split(":")
#         suff = " " + str(start) + ",\n"
#         lines[car_line] = pre + ":" + suff
#     with open(yaml_file, 'w') as f:
#         for line in lines:
#             f.write(line)
#     os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_contra.yaml --fix_random_seed --extra_tag second_car_clocs_contra_weight' + str(start)+
#               ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')
#     start *= increase
# os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs.yaml --fix_random_seed --extra_tag epoch10' + \
#         ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')


loops = 3
while(loops>0):
    # seed = random.randint(1, 10000)
    seed = 666
    os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs.yaml --fix_random_seed ' + str(seed) + ' --extra_tag retainnet_epoch10_no_gt_sampling_ours_no_pad_seed' + str(seed) +\
        ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')
    os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_fusion_aug.yaml --fix_random_seed ' + str(seed) + ' --extra_tag retainnet_epoch10_gt_sampling_ours_no_pad_seed' + str(seed) +\
        ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')

    loops-=1
