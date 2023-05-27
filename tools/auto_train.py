import os
from decimal import Decimal
# os.system('python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --extra_tag origin')
# os.system('python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --extra_tag origin')
# os.system('python train.py --cfg_file cfgs/kitti_models/PartA2_free.yaml --extra_tag origin')
# os.system('python train.py --cfg_file cfgs/kitti_models/PartA2.yaml --extra_tag origin')

yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_new_assigner.yaml"
car_line = 69

target = Decimal('-0.3')
start = Decimal('-0.7')
increase = Decimal('0.05')
while start <= target:
    with open(yaml_file, 'r') as f:
        lines = f.readlines()
        pre, suff = lines[car_line].split(":")
        suff = " " + str(start) + ",\n"
        lines[car_line] = pre + ":" + suff
    with open(yaml_file, 'w') as f:
        for line in lines:
            f.write(line)
    os.system('python train.py --cfg_file cfgs/kitti_models/second_new_assigner.yaml --extra_tag new_assigner_cyc_match' + str(start))
    start += increase
    
