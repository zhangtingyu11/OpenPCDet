import os
from decimal import Decimal
import random

SECOND_CLOCS_CONTRA_FUSION_AUG_FILE = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml"
SECOND_CLOCS_FILE = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs.yaml"
SECOND_PRETRAINED_MODEL = "../output/kitti_models/second_car/origin/ckpt/checkpoint_epoch_80.pth"
SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE = "cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml"

def change_config_file(config_file, line_idx, change_component, need_comma=False):
    """修改配置文件中的某个信息

    Args:
        config_file (_type_): 配置文件的路径
        line_idx (_type_): 需要修改的行数
        change_component (_type_): 需要修改成什么内容
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()
        modify_line = lines[line_idx]
        splits = modify_line.split(': ')
        if need_comma:
            splits[-1] = str(change_component) + ',\n'
        else:
            splits[-1] = str(change_component) + '\n'
        lines[line_idx] = ': '.join(splits)
        
    with open(config_file, 'w') as f:
        for line in lines:
            f.write(line)
        
def auto_train_command(cfg_file, random_seed, tag, pretrained_model=None):
    """填充训练的指令
    
    Args:
        cfg_file (_type_): 配置文件
        random_seed (_type_): 随机种子
        tag (_type_): 训练时的标签
        pretrained_model (_type_, optional): 预训练模型. Defaults to None.
    """
    if pretrained_model is None:
        os.system("python train.py --cfg_file {} --fix_random_seed {} --extra_tag {}".format(
            cfg_file, random_seed, tag
        ))
    else:
        os.system("python train.py --cfg_file {} --fix_random_seed {} --extra_tag {} --pretrained_model {}".format(
            cfg_file, random_seed, tag, pretrained_model
        ))

def influence_of_detector2d():
    """测试不同准确度的2D目标检测器对结果的影响
    """
    scores = [[0, 0.2], [0.6, 1.0], [0.9, 1.0]]
    propotion_line_idx = 69
    low_score_line_idx = 70
    high_score_line_idx = 71
    propotions = [0.5, 1]
    for propotion in propotions:
        change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, propotion_line_idx, propotion)
        for low_score, high_score in scores:
            change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, low_score_line_idx, low_score)
            change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, high_score_line_idx, high_score)
            extra_tag = "origin_high{}_low{}".format(high_score, low_score)   
            auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, extra_tag, SECOND_PRETRAINED_MODEL)

#TODO
def choose_contra_weight():
    """测试不同的对比学习loss权重
    """
    contra_weights = ['0.001', '0.01', '0.1', '0.2', '0.3,', '0.4', '0.5', '0.6', '0.7', '0.8', '1.0']
    line_idx = 149
    for contra_weight in contra_weights:
        change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, line_idx, contra_weight, need_comma=True)
        extra_tag = "origin_contraweight{}".format(contra_weight)
        auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, extra_tag, SECOND_PRETRAINED_MODEL)

def choose_pos_neg_thresh():
    """测试不同的pos neg阈值
    """
    clos_pos_iou_threshs = ['0.5', '0.55', '0.6', '0.65', '0.7']
    pos_iou_threshs_line = 122
    clos_neg_iou_threshs = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35']
    neg_iou_threshs_line = 123
    for pos_thr in clos_pos_iou_threshs:
        change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, pos_iou_threshs_line, pos_thr)
        for neg_thr in clos_neg_iou_threshs:
            change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, neg_iou_threshs_line, neg_thr)
            extra_tag = "origin_pos{}_neg{}".format(pos_thr, neg_thr)
            auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, extra_tag, SECOND_PRETRAINED_MODEL)
            
def choose_iou_thresh():
    """测试不同的IOU_THRESH
    """
    iou_threshs = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7']
    line_idx = 121
    for iou_thresh in iou_threshs:
        change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, line_idx, iou_thresh)
        extra_tag = "origin_iouthresh{}".format(iou_thresh)
        auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, extra_tag, SECOND_PRETRAINED_MODEL)
        
        
def choose_contra_match_iou():
    """测试不同的CONTRA_MATCH_IOU
    """
    contra_match_ious = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.55', '0.6', '0.65', '0.7', '0.8', '0.9']
    line_idx = 120
    for contra_match_iou in contra_match_ious:
        change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, line_idx, contra_match_iou)
        extra_tag = "origin_contrathresh{}".format(contra_match_iou)
        auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, extra_tag, SECOND_PRETRAINED_MODEL)
        
#TODO
def choose_random_seed():
    loops = 10
    for _ in range(loops):
        seed = random.randint(1, 10000)
        extra_tag = "origin_seed{}".format(seed)
        auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, seed, extra_tag, SECOND_CLOCS_CONTRA_FUSION_AUG_FILE)
    
def train_with_and_wo_aug():
    auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, "use_la_mgs_epoch10", SECOND_PRETRAINED_MODEL)
    change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, 117, "5")
    change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, 126, "True")
    change_config_file(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE, 127, "False")
    auto_train_command(SECOND_CLOCS_CONTRA_FUSION_AUG_FILE_RELATIVE, 666, "use_cl_mgs_epoch10", SECOND_PRETRAINED_MODEL)
    
    

if __name__=="__main__":
    # choose_contra_weight()
    train_with_and_wo_aug()
