def analyze_log_file(log_file):
    #* moderate, hard, easy
    car_11_result = [0,0,0]
    car_40_result = [0,0,0]
    pedestrian_11_result = [0,0,0]
    pedestrian_40_result = [0,0,0]
    cyclist_11_result = [0,0,0]
    cyclist_40_result = [0,0,0]
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if(line.endswith("Car AP@0.70, 0.70, 0.70:\n")):
                cur_3d_line = lines[idx+3]
                split_line = cur_3d_line.split(" ")
                cur_car_easy_11 = float(split_line[3][3:-1])
                cur_car_moderate_11 = float(split_line[4][:-1])
                cur_car_hard_11 = float(split_line[5][:-2])
                cur_car_11_result = [cur_car_moderate_11, cur_car_hard_11, cur_car_easy_11]
                if(cur_car_11_result > car_11_result):
                    car_11_result = cur_car_11_result
            elif(line.endswith("Car AP_R40@0.70, 0.70, 0.70:\n")):
                cur_3d_line = lines[idx+3]
                split_line = cur_3d_line.split(" ")
                cur_car_easy_40 = float(split_line[3][3:-1])
                cur_car_moderate_40 = float(split_line[4][:-1])
                cur_car_hard_40 = float(split_line[5][:-2])
                cur_car_40_result = [cur_car_moderate_40, cur_car_hard_40, cur_car_easy_40]
                if(cur_car_40_result > car_40_result):
                    car_40_result = cur_car_40_result
            elif(line.endswith("Pedestrian AP@0.50, 0.50, 0.50:\n")):
                cur_3d_line = lines[idx+3]
                split_line = cur_3d_line.split(" ")
                cur_pedestrian_easy_11 = float(split_line[3][3:-1])
                cur_pedestrian_moderate_11 = float(split_line[4][:-1])
                cur_pedestrian_hard_11 = float(split_line[5][:-2])
                cur_pedestrian_11_result = [cur_pedestrian_moderate_11, cur_pedestrian_hard_11, cur_pedestrian_easy_11]
                if(cur_pedestrian_11_result > pedestrian_11_result):
                    pedestrian_11_result = cur_pedestrian_11_result
            elif(line.endswith("Pedestrian AP_R40@0.50, 0.50, 0.50:\n")):
                cur_3d_line = lines[idx+3]
                split_line = cur_3d_line.split(" ")
                cur_pedestrian_easy_40 = float(split_line[3][3:-1])
                cur_pedestrian_moderate_40 = float(split_line[4][:-1])
                cur_pedestrian_hard_40 = float(split_line[5][:-2])
                cur_pedestrian_40_result = [cur_pedestrian_moderate_40, cur_pedestrian_hard_40, cur_pedestrian_easy_40]
                if(cur_pedestrian_40_result > pedestrian_40_result):
                    pedestrian_40_result = cur_pedestrian_40_result
            elif(line.endswith("Cyclist AP@0.50, 0.50, 0.50:\n")):
                cur_3d_line = lines[idx+3]
                split_line = cur_3d_line.split(" ")
                cur_cyclist_easy_11 = float(split_line[3][3:-1])
                cur_cyclist_moderate_11 = float(split_line[4][:-1])
                cur_cyclist_hard_11 = float(split_line[5][:-2])
                cur_cyclist_11_result = [cur_cyclist_moderate_11, cur_cyclist_hard_11, cur_cyclist_easy_11]
                if(cur_cyclist_11_result > cyclist_11_result):
                    cyclist_11_result = cur_cyclist_11_result
            elif(line.endswith("Cyclist AP_R40@0.50, 0.50, 0.50:\n")):
                cur_3d_line = lines[idx+3]
                split_line = cur_3d_line.split(" ")
                cur_cyclist_easy_40 = float(split_line[3][3:-1])
                cur_cyclist_moderate_40 = float(split_line[4][:-1])
                cur_cyclist_hard_40 = float(split_line[5][:-2])
                cur_cyclist_40_result = [cur_cyclist_moderate_40, cur_cyclist_hard_40, cur_cyclist_easy_40]
                if(cur_cyclist_40_result > cyclist_40_result):
                    cyclist_40_result = cur_cyclist_40_result
    
    print("car_AP11", car_11_result[2], car_11_result[0], car_11_result[1])
    print("car_AP40", car_40_result[2], car_40_result[0], car_40_result[1])
    print("pedestrian_AP11", pedestrian_11_result[2], pedestrian_11_result[0], pedestrian_11_result[1])
    print("pedestrian_AP40", pedestrian_40_result[2], pedestrian_40_result[0], pedestrian_40_result[1])
    print("cyclist_AP11", cyclist_11_result[2], cyclist_11_result[0], cyclist_11_result[1])
    print("cyclist_AP40", cyclist_40_result[2], cyclist_40_result[0], cyclist_40_result[1])
    
        
    
                

if __name__ == "__main__":
    analyze_log_file("/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/3dssd_dsasa_use_all_backbone/origin/log_train_20230211-002242.txt")