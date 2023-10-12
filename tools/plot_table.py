import os
import matplotlib.pyplot as plt

DIR_PATH = "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second_car_clocs_contra_fusion_aug"

def plot_pos_neg_thr_test():
    info_dict = {}
    pos_set = set()
    neg_set = set()
    for root, dirs, files in os.walk(DIR_PATH):
        suff = root.split('/')[-1]
        if("origin_pos" in suff):
            splits = suff.split('_')
            pos_thr = float(splits[1][3:])
            neg_thr = float(splits[2][3:])
            pos_set.add(pos_thr)
            neg_set.add(neg_thr)
            for filename in files:
                if filename.endswith('.log'):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for idx, line in enumerate(lines):
                            if("Car AP_R40@0.70, 0.70, 0.70:" in line):
                                info = lines[idx+3]
                                data = info.strip().split(':')[-1]
                                data_str = data.split(', ')
                                data_float = list(map(float, data_str))
                                data_float = list(map(lambda x: round(x, 2), data_float))
                                easy, mod, hard = data_float
                                total = (easy+mod+hard)/3
                                total = round(total, 2)
                                break
                    break
            info_dict[(pos_thr, neg_thr)] = (easy, mod, hard, total)

    pos_list = sorted(pos_set)
    neg_list = sorted(neg_set)
    plt.figure(figsize=(20,8))
    data_total = []
    for i in range(len(pos_list)):
        row = []
        for j in range(len(neg_list)):
            row.append(info_dict[(pos_list[i], neg_list[j])])
        data_total.append(row)
    plt.figure(figsize=(20, 8))
    tab = plt.table(cellText=data_total, 
                colLabels=neg_list, 
                rowLabels=pos_list,
                loc='center', 
                cellLoc='center',
                rowLoc='center')
    tab.scale(1,2) 
    plt.axis('off')
    plt.show()
    
def plot_iou_thresh_test():
    info_dict = {}
    iou_thr_set = set()
    for root, dirs, files in os.walk(DIR_PATH):
        suff = root.split('/')[-1]
        if("iouthresh" in suff):
            splits = suff.split('_')
            iou_thr = float(splits[1][9:])
            iou_thr_set.add(iou_thr)
            for filename in files:
                if filename.endswith('.log'):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for idx, line in enumerate(lines):
                            if("Car AP_R40@0.70, 0.70, 0.70:" in line):
                                info = lines[idx+3]
                                data = info.strip().split(':')[-1]
                                data_str = data.split(', ')
                                data_float = list(map(float, data_str))
                                data_float = list(map(lambda x: round(x, 2), data_float))
                                easy, mod, hard = data_float
                                total = (easy+mod+hard)/3
                                total = round(total, 2)
                                break
                    break
            info_dict[iou_thr] = (easy, mod, hard, total)
    
    plt.figure(figsize=(20,8))
    iou_thr_list = sorted(iou_thr_set)
    data_total = []
    for iou_thr in iou_thr_list:
        data_total.append(info_dict[iou_thr])
    plt.figure(figsize=(20, 8))
    tab = plt.table(cellText=data_total, 
                rowLabels=iou_thr_list,
                loc='center', 
                cellLoc='center',
                rowLoc='center')
    tab.scale(1,2) 
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    plot_iou_thresh_test()

        
                            