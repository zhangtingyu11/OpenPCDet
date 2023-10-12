portion = 0.8
trainval_file = '/home/zty/Project/DeepLearning/OpenPCDet/data/kitti/ImageSets/trainval.txt'
selected_file_name = 'train.txt'
unselected_file_name = 'val.txt'
import random
# 打开输入文件和输出文件
with open(trainval_file, "r") as input_file, open(selected_file_name, "w") as selected_file, open(unselected_file_name, "w") as unselected_file:
    # 读取输入文件的所有行
    lines = input_file.readlines()
    
    # 设置要选择的行数
    n = len(lines)
    num_selected = int(n*portion) # 选择前5行，你可以根据需要更改
    
    selected_indices = random.sample(range(len(lines)), num_selected)
    
    # 将选中的索引按升序排序，以保留原有的顺序
    selected_indices.sort()
    
    # 写入选择的行到selected_file
    for index in selected_indices:
        selected_file.write(lines[index])
    
    # 写入未选择的行到unselected_file
    for i, line in enumerate(lines):
        if i not in selected_indices:
            unselected_file.write(line)