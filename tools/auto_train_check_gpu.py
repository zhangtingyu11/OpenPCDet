import subprocess
import os
import time

def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpu_memory = [int(x.split()[0]) for i, x in enumerate(output.strip().split('\n')[1:]) if i == 0]
    return gpu_memory

def auto_train():
    #! 训练最初版本的clocs
    os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs.yaml --fix_random_seed 666 --extra_tag origin_epoch10_seed666' +\
            ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')
    
    #! 添加Contra和添加LA的版本
    #* USE_GT, VAL_ACTIVATED, PROPOTION, LOWER_SCORE, HIGHER_SCORE, USE_CONTRA, USE_LA
    no_gt_configs = [[False, False, 0, 0, 0, True, False],
                     [False, False, 0, 0, 0, False, True]]
    no_gt_lines = [66, 67, 68, 69, 70, 122, 123]
    tags = ["use_contra", "use_la"]
    yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra.yaml"
    for i, config in enumerate(no_gt_configs):
        tag = tags[i]
        with open(yaml_file, 'r') as f:
            lines = f.readlines()
            for idx, line_idx in enumerate(no_gt_lines):
                config_line = lines[line_idx]
                splits = config_line.split(': ')
                splits[-1] = str(config[idx])+'\n'
                lines[line_idx] = ': '.join(splits)
        with open(yaml_file, 'w') as f:
            for line in lines:
                f.write(line)
        os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_contra.yaml --fix_random_seed 666 --extra_tag ' + tag + '_epoch10_no_gt_sampling_ours_filter_no_pad_seed666' +\
            ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')   
            
        
    #! 使用gt的版本
    configs = [[False, False, 0, 0, 0, False, False],
                [False, False, 0, 0, 0, True, True],
                [True, False, 0.5, 0.6, 1.0, True, True],
                [True, False, 0.5, 0.8, 1.0, True, True],
                [True, False, 1.0, 0.6, 1.0, True, True],
                [True, False, 1.0, 0.8, 1.0, True, True]]
    line_idxs = [67, 68, 69, 70, 71, 123, 124]
    tags = ["use_gt_sampling",
            "use_all",
            "use_all_0.5_0.6_1.0",
            "use_all_0.5_0.8_1.0"
            "use_all_1.0_0.6_1.0",
            "use_all_1.0_0.8_1.0"]
    yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml"
    for i, config in enumerate(configs):
        tag = tags[i]
        with open(yaml_file, 'r') as f:
            lines = f.readlines()
            for idx, line_idx in enumerate(line_idxs):
                config_line = lines[line_idx]
                splits = config_line.split(': ')
                splits[-1] = str(config[idx])+'\n'
                lines[line_idx] = ': '.join(splits)
        with open(yaml_file, 'w') as f:
            for line in lines:
                f.write(line)
        os.system('python train.py --cfg_file cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml --fix_random_seed 666 --extra_tag ' + tag + '_epoch10_seed666' +\
            ' --pretrained_model ../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth')  
        
if __name__ == "__main__":
    gpu_memory = get_gpu_memory()[0]
    while(gpu_memory >100):
        time.sleep(600)
        gpu_memory = get_gpu_memory()[0]
    auto_train()
    
