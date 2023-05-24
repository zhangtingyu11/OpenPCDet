import os
os.system('python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --extra_tag origin')
os.system('python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --extra_tag origin')
# os.system('python train.py --cfg_file cfgs/kitti_models/PartA2_free.yaml --extra_tag origin')
# os.system('python train.py --cfg_file cfgs/kitti_models/PartA2.yaml --extra_tag origin')
