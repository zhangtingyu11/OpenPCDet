import os
import matplotlib.pyplot as plt

DIR_PATH = "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second_car_clocs_contra_fusion_aug"

def plot_dim2_data(dir_path, filter_str, start_idx1, start_idx2):
    info_dict = {}
    dim1_data_set = set()
    dim2_data_set = set()
    for root, _, files in os.walk(dir_path):
        suff = root.split('/')[-1]
        if(filter_str in suff):
            splits = suff.split('_')
            num1 = float(splits[1][start_idx1:])
            num2 = float(splits[2][start_idx2:])
            dim1_data_set.add(num1)
            dim2_data_set.add(num2)
            easy, mod, hard, total = extract_data(root, files)
            info_dict[(num1, num2)] = (easy, mod, hard, total)

    info1_list = sorted(dim1_data_set)
    info2_list = sorted(dim2_data_set)
    data_total = []
    for i in range(len(info1_list)):
        row = []
        for j in range(len(info2_list)):
            row.append(info_dict[(info1_list[i], info2_list[j])])
        data_total.append(row)
    plt.figure(figsize=(20,8))
    tab = plt.table(cellText=data_total, 
                colLabels=info2_list, 
                rowLabels=info1_list,
                loc='center', 
                cellLoc='center',
                rowLoc='center')
    tab.scale(1,2) 
    plt.axis('off')
    plt.show()
    
def extract_data(root_path, files):
    b_easy=b_mod=b_hard=b_total=0
    for filename in files:
        if filename.endswith('.log'):
            file_path = os.path.join(root_path, filename)
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
                        if(total > b_total):
                            b_total = total
                            b_easy = easy
                            b_mod = mod
                            b_hard = hard
    return b_easy, b_mod, b_hard, b_total

def plot_dim1_data(dir_path, filter_str, start_idx):
    info_dict = {}
    info_set = set()
    for root, _, files in os.walk(dir_path):
        suff = root.split('/')[-1]
        if filter_str in suff:
            splits = suff.split('_')
            num = float(splits[1][start_idx:])
            info_set.add(num)
            easy, mod, hard, total = extract_data(root, files)
            info_dict[num] = (easy, mod, hard, total)
    plt.figure(figsize=(20,8))
    info_list = sorted(info_set)
    data_total = []
    for info in info_list:
        data_total.append(info_dict[info])
    tab = plt.table(cellText=data_total, 
                rowLabels=info_list,
                loc='center', 
                cellLoc='center',
                rowLoc='center')
    tab.scale(1,2) 
    plt.axis('off')
    plt.show()

    
if __name__ == "__main__":
    plot_dim1_data(DIR_PATH, "origin_contraweight", 12)
    # plot_dim2_data(DIR_PATH, "origin_pos", 3, 3)

        
                            