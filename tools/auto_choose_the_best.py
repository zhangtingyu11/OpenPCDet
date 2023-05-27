import os
import logging
from enum import IntEnum

class Mode(IntEnum):
    MEAN = 0
    SINGLE_CLASS = 1
    
class AutoFindBest:
    def __init__(self, mode = 0) -> None:
        self.mode = mode
        
    def find_single_file_best(self, filename):
        if(self.mode == int(Mode.MEAN)):
            return self.find_single_file_best_mean(filename)  
            
    def find_directory_best(self, search_paths):
        if(self.mode == Mode.MEAN):
            self.find_directory_best_mean(search_paths)
 
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        self._mode = value
        
class AutoFindBestKitti(AutoFindBest):
    def __init__(self, mode=0) -> None:
        super().__init__(mode)
        self.car_ap11_line = "Car AP@0.70, 0.70, 0.70:\n"
        self.car_ap40_line = "Car AP_R40@0.70, 0.70, 0.70:\n"
        self.ped_ap11_line = "Pedestrian AP@0.50, 0.50, 0.50:\n"
        self.ped_ap40_line = "Pedestrian AP_R40@0.50, 0.50, 0.50:\n"
        self.cyc_ap11_line = "Cyclist AP@0.50, 0.50, 0.50:\n"
        self.cyc_ap40_line = "Cyclist AP_R40@0.50, 0.50, 0.50:\n"
        
    def find_single_file_best_mean(self, filename):
        #* 记录最好的一次结果
        best_mean_ap11 = (0, 0, 0)
        best_mean_ap40 = (0, 0, 0)
        
        #* 计算总共有多少个类别
        car_enable, ped_enable, cyc_enbale = False, False, False
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.endswith(self.car_ap11_line)):
                    car_enable = True
                elif(line.endswith(self.ped_ap11_line)):
                    ped_enable = True
                elif(line.endswith(self.cyc_ap11_line)):
                    cyc_enbale = True
                    
        classes_num = car_enable+ped_enable+cyc_enbale
        
        first = None
        #* 判断第一个存在的类别
        if(car_enable):
            first = self.car_ap11_line
        else:
            if(ped_enable):
                first = self.ped_ap11_line
            else:
                if(cyc_enbale):
                    first = self.cyc_ap11_line
        if(first is None):
            return best_mean_ap11, best_mean_ap40
        def reset():
            return (0, 0, 0), (0, 0, 0), (0, 0, 0)

        car_ap11_result, ped_ap11_result, cyc_ap11_result = reset()
        car_ap40_result, ped_ap40_result, cyc_ap40_result = reset()
        with open(filename, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if(line.endswith(first)):
                    ap_11_result = tuple(map(sum, zip(car_ap11_result, ped_ap11_result, cyc_ap11_result)))
                    ap_40_result = tuple(map(sum, zip(car_ap40_result, ped_ap40_result, cyc_ap40_result)))
                    if(sum(ap_11_result) >sum(best_mean_ap11)):
                        best_mean_ap11 = ap_11_result
                    if(sum(ap_40_result) >sum(best_mean_ap40)):
                        best_mean_ap40 = ap_40_result
                if(line.endswith(self.car_ap11_line)):
                    car_ap11_result = lines[idx+3]
                    _, suff = car_ap11_result.split(':')
                    car_ap11_result = suff.split(', ')
                    car_ap11_result = tuple(map(float, car_ap11_result))
                elif(line.endswith(self.car_ap40_line)):
                    car_ap40_result = lines[idx+3]
                    _, suff = car_ap40_result.split(':')
                    car_ap40_result = suff.split(', ')
                    car_ap40_result = tuple(map(float, car_ap40_result))
                elif(line.endswith(self.ped_ap11_line)):
                    ped_ap11_result = lines[idx+3]
                    _, suff = ped_ap11_result.split(':')
                    ped_ap11_result = suff.split(', ')
                    ped_ap11_result = tuple(map(float, ped_ap11_result))
                elif(line.endswith(self.ped_ap40_line)):
                    ped_ap40_result = lines[idx+3]
                    _, suff = ped_ap40_result.split(':')
                    ped_ap40_result = suff.split(', ')
                    ped_ap40_result = tuple(map(float, ped_ap40_result))
                elif(line.endswith(self.cyc_ap11_line)):
                    cyc_ap11_result = lines[idx+3]
                    _, suff = cyc_ap11_result.split(':')
                    cyc_ap11_result = suff.split(', ')
                    cyc_ap11_result = tuple(map(float, cyc_ap11_result))
                elif(line.endswith(self.cyc_ap40_line)):
                    cyc_ap40_result = lines[idx+3]
                    _, suff = cyc_ap40_result.split(':')
                    cyc_ap40_result = suff.split(', ')
                    cyc_ap40_result = tuple(map(float, cyc_ap40_result))
        ap_11_result = tuple(map(sum, zip(car_ap11_result, ped_ap11_result, cyc_ap11_result)))
        ap_40_result = tuple(map(sum, zip(car_ap40_result, ped_ap40_result, cyc_ap40_result)))
        if(sum(ap_11_result) >sum(best_mean_ap11)):
            best_mean_ap11 = ap_11_result
        if(sum(ap_40_result) >sum(best_mean_ap40)):
            best_mean_ap40 = ap_40_result
        if(classes_num == 0):
            return best_mean_ap11, best_mean_ap40
        else:
            return tuple(map(lambda x:x/3, best_mean_ap11)), tuple(map(lambda x:x/3, best_mean_ap40))
        
    def find_directory_best_mean(self, search_paths):
        best_ap11 = (0, 0, 0)
        best_ap40 = (0, 0, 0)
        best_ap11_file = None
        best_ap40_file = None
        for search_path in search_paths:
            for root, dir, files in os.walk(search_path):
                for file in files:
                    if(file.endswith('log')):
                        cur_ap11, cur_ap40 = self.find_single_file_best_mean(os.path.join(root, file))
                        if(sum(cur_ap11) > sum(best_ap11)):
                            best_ap11 = cur_ap11
                            best_ap11_file = os.path.join(root, file)
                        if(sum(cur_ap40) > sum(best_ap40)):
                            best_ap40 = cur_ap40
                            best_ap40_file = os.path.join(root, file)
        logging.info("best_mean_ap11:{}".format(best_ap11))   
        logging.info("best_mean_ap11_file:{}".format(best_ap11_file))    
        logging.info("best_mean_ap40:{}".format(best_ap40))   
        logging.info("best_mean_ap40_file:{}".format(best_ap40_file))  
        logging.info("-------------------------------------------------------------------------")    
          

            
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    search_paths = ["/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second_new_assigner/",
                    "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second/"]
    baseline_path = "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second/origin/train_20230522-114220.log"
    
    instance = AutoFindBestKitti()
    base_ap11, base_ap40 = instance.find_single_file_best(baseline_path)
    logging.info("Baseline AP11:{}".format(base_ap11))
    logging.info("Baseline AP40:{}".format(base_ap40))
    instance.find_directory_best(search_paths)
    
    
    

                        
    

