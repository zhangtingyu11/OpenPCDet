import os

search_path = "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second_new_assigner/"
base_file = "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second/origin/train_20230522-114220.log"

#TODO 可以把这个优化一下，变成

result_dict = {
    'Car':{},
    'Pedestrian':{},
    'Cyclist':{}
}

def div3(num):
    return num/3

def judge_one_file_avg_ap(filename):
    best_ap11 = (0, 0, 0)
    best_ap40 = (0, 0, 0)
    with open(cur_search_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if(line.endswith('Car AP@0.70, 0.70, 0.70:\n')):
                car_ap11_result = lines[idx+3]
                pre, suff = car_ap11_result.split(':')
                car_ap11_result = suff.split(', ')
                car_ap11_result = tuple(map(float, car_ap11_result))
                
                car_ap40_result = lines[idx+8]
                pre, suff = car_ap40_result.split(':')
                car_ap40_result = suff.split(', ')
                car_ap40_result = tuple(map(float, car_ap40_result))

                ped_ap11_result = lines[idx+13]
                pre, suff = ped_ap11_result.split(':')
                ped_ap11_result = suff.split(', ')
                ped_ap11_result = tuple(map(float, ped_ap11_result))

                ped_ap40_result = lines[idx+18]
                pre, suff = ped_ap40_result.split(':')
                ped_ap40_result = suff.split(', ')
                ped_ap40_result = tuple(map(float, ped_ap40_result))

                cyc_ap11_result = lines[idx+23]
                pre, suff = cyc_ap11_result.split(':')
                cyc_ap11_result = suff.split(', ')
                cyc_ap11_result = tuple(map(float, cyc_ap11_result))

                cyc_ap40_result = lines[idx+28]
                pre, suff = cyc_ap40_result.split(':')
                cyc_ap40_result = suff.split(', ')
                cyc_ap40_result = tuple(map(float, cyc_ap40_result))
                
                ap_11_result = tuple(map(sum, zip(car_ap11_result, ped_ap11_result, cyc_ap11_result)))
                ap_40_result = tuple(map(sum, zip(car_ap40_result, ped_ap40_result, cyc_ap40_result)))
                if(sum(ap_11_result) > sum(best_ap11)):
                    best_ap11 = ap_11_result
                if(sum(ap_40_result) > sum(best_ap40)):
                    best_ap40 = ap_40_result
    return best_ap11, best_ap40

for sub_dir in os.listdir(search_path):
    sub_search_path = os.path.join(search_path, sub_dir)
    for file in os.listdir(sub_search_path):
        if(file.endswith('.log')):
            cur_search_path = os.path.join(sub_search_path, file)
            with open(cur_search_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if(line.endswith('Car AP@0.70, 0.70, 0.70:\n')):
                        car_ap11_result = lines[idx+3]
                        pre, suff = car_ap11_result.split(':')
                        car_ap11_result = suff.split(', ')
                        car_ap11_result = tuple(map(float, car_ap11_result))
                        if car_ap11_result > result_dict['Car'].get('Best_Car_AP11', (0, 0, 0)):
                            result_dict['Car']['Best_Car_AP11'] = car_ap11_result
                            result_dict['Car']['Best_Car_AP11_File'] = sub_dir
                    elif(line.endswith('Car AP_R40@0.70, 0.70, 0.70:\n')):
                        car_ap40_result = lines[idx+3]
                        pre, suff = car_ap40_result.split(':')
                        car_ap40_result = suff.split(', ')
                        car_ap40_result = tuple(map(float, car_ap40_result))
                        if car_ap40_result > result_dict['Car'].get('Best_Car_AP40', (0, 0, 0)):
                            result_dict['Car']['Best_Car_AP40'] = car_ap40_result
                            result_dict['Car']['Best_Car_AP40_File'] = sub_dir
                    elif(line.endswith('Pedestrian AP@0.50, 0.50, 0.50:\n')):
                        ped_ap11_result = lines[idx+3]
                        pre, suff = ped_ap11_result.split(':')
                        ped_ap11_result = suff.split(', ')
                        ped_ap11_result = tuple(map(float, ped_ap11_result))
                        if ped_ap11_result > result_dict['Pedestrian'].get('Best_Ped_AP11', (0, 0, 0)):
                            result_dict['Pedestrian']['Best_Ped_AP11'] = ped_ap11_result
                            result_dict['Pedestrian']['Best_Ped_AP11_File'] = sub_dir
                    elif(line.endswith('Pedestrian AP_R40@0.50, 0.50, 0.50:\n')):
                        ped_ap40_result = lines[idx+3]
                        pre, suff = ped_ap40_result.split(':')
                        ped_ap40_result = suff.split(', ')
                        ped_ap40_result = tuple(map(float, ped_ap40_result))
                        if ped_ap40_result > result_dict['Pedestrian'].get('Best_Ped_AP40', (0, 0, 0)):
                            result_dict['Pedestrian']['Best_Ped_AP40'] = ped_ap40_result
                            result_dict['Pedestrian']['Best_Ped_AP40_File'] = sub_dir
                    elif(line.endswith('Cyclist AP@0.50, 0.50, 0.50:\n')):
                        cyc_ap11_result = lines[idx+3]
                        pre, suff = cyc_ap11_result.split(':')
                        cyc_ap11_result = suff.split(', ')
                        cyc_ap11_result = tuple(map(float, cyc_ap11_result))
                        if cyc_ap11_result > result_dict['Cyclist'].get('Best_Ped_AP11', (0, 0, 0)):
                            result_dict['Cyclist']['Best_Cyc_AP11'] = cyc_ap11_result
                            result_dict['Cyclist']['Best_Cyc_AP11_File'] = sub_dir
                    elif(line.endswith('Cyclist AP_R40@0.50, 0.50, 0.50:\n')):
                        cyc_ap40_result = lines[idx+3]
                        pre, suff = cyc_ap40_result.split(':')
                        cyc_ap40_result = suff.split(', ')
                        cyc_ap40_result = tuple(map(float, cyc_ap40_result))
                        if cyc_ap40_result > result_dict['Cyclist'].get('Best_Cyc_AP40', (0, 0, 0)):
                            result_dict['Cyclist']['Best_Cyc_AP40'] = cyc_ap40_result
                            result_dict['Cyclist']['Best_Cyc_AP40_File'] = sub_dir
base_ap11, base_ap40 = judge_one_file_avg_ap(base_file)
base_ap11 = tuple(map(div3, base_ap11)) 
base_ap40 = tuple(map(div3, base_ap40)) 
print("baseline_ap11:{}".format(base_ap11))
print("baseline_ap40:{}".format(base_ap40))
print("-------------------------------------------------------------------")

for class_name, res in result_dict.items():
    for key, val in res.items():
        print('{}:{}'.format(key, val))
        
print("-------------------------------------------------------------------")

best_ap11 = (0, 0, 0)
best_ap40 = (0, 0, 0)
best_ap11_file = None
best_ap40_file = None
for sub_dir in os.listdir(search_path):
    sub_search_path = os.path.join(search_path, sub_dir)
    for file in os.listdir(sub_search_path):
        if(file.endswith('.log')):
            cur_search_path = os.path.join(sub_search_path, file)
            cur_ap11, cur_ap40 = judge_one_file_avg_ap(cur_search_path)
            if(sum(cur_ap11) > sum(best_ap11)):
                best_ap11 = cur_ap11
                best_ap11_file = sub_dir
            if(sum(cur_ap40) > sum(best_ap40)):
                best_ap40 = cur_ap40
                best_ap40_file = sub_dir

print("best ap11:{}".format(tuple(map(div3, best_ap11))))
print("best_ap11_file:{}".format(best_ap11_file))
print("best ap40:{}".format(tuple(map(div3, best_ap40))))
print("best_ap40_file:{}".format(best_ap40_file))


                        

                        
    

