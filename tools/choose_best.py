import os
def choose_best_KITTI_CAR(dirname):
    b_total = 0
    b_file = None
    b_easy = 0
    b_mod = 0
    b_hard = 0
    files = tranverse(dirname)
    for filename in files:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if 'Car AP_R40@0.70, 0.70, 0.70:' in line:
                    result_3d = lines[idx+3].strip()
                    data = result_3d.split(':')[1]
                    stat = data.split(', ')
                    stat = list(map(float, stat))
                    easy, mod, hard = stat
                    total = easy+hard+mod
                    if(total > b_total):
                        b_total = total
                        b_easy = easy
                        b_mod = mod
                        b_hard = hard
                        b_file = filename
    return b_total/3, b_easy, b_mod, b_hard, b_file

def tranverse(dirname, suff = ".log"):
    paths = []
    for foldername, subfolders, filenames in os.walk(dirname):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if(file_path.endswith(suff)):
                paths.append(file_path)
    return paths

if __name__ == '__main__':
    print(choose_best_KITTI_CAR('/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/voxel_rcnn_car'))