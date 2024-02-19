import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import re
import os
import numpy as np
yaml_file = "/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml"
change_line_idx = 160
directory = "/home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second_car_clocs_contra_fusion_aug/default/eval/epoch_10/val/default"
def change_yaml_name(yaml_file, threshold):
    with open(yaml_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'SCORE_THRESH' in line:
                log_dir_content = lines[idx]
                splits = log_dir_content.split(': ')
                splits[-1] = str(threshold) + '\n'
                log_dir_content = ': '.join(splits)
                lines[idx] = log_dir_content
    with open(yaml_file, 'w') as f:
        f.writelines(lines)
        
def send_email(easy, mod, hard, map, threshold):
    # 发件人和收件人信息
    sender_email = "18013933973@163.com"
    receiver_email = "18013933973@163.com"
    password = "SNOYAHKUJNPWATEF"

    # 邮件内容
    subject = "C-CLOCS阈值的表现"
    body = "在阈值为{}下, 当前各个难度的AP为({}, {}, {}), mAP为{}".format(threshold, easy, mod, hard, map)
    
    # 创建 MIMEText 对象
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = Header(subject, "utf-8")
    message["From"] = sender_email
    message["To"] = receiver_email
    # 连接到网易邮箱 SMTP 服务器
    with smtplib.SMTP("smtp.163.com", 25) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())        

result = []
best_mean = 0
for threshold in np.arange(0.1, 0.9, 0.01):
    change_yaml_name(yaml_file, threshold)
    os.system('python test.py --cfg_file cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml --batch_size 8 --ckpt /home/zty/Project/DeepLearning/OpenPCDet/output/kitti_models/second_car_clocs_contra_fusion_aug/batchsize8_weightinit_epoch10_voxelrcnn_strategy_no_contra/ckpt/checkpoint_epoch_10.pth')
    filenames = os.listdir(directory)
    filtered_filenames = [filename for filename in filenames if filename.endswith('.txt')]
    filtered_filenames.sort()
    current_filename = filtered_filenames[-1]
    with open(os.path.join(directory, current_filename), 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "Car AP_R40@0.70, 0.70, 0.70:" in line:
                content = lines[idx+3]
                sp = content.split(":")[-1]
                easy, mod, hard = sp.strip().split(', ')
                easy = float(easy)
                mod = float(mod)
                hard = float(hard)
                mean = (easy + mod + hard)/3
                if mean > best_mean:
                    best_mean = mean
                    result = [easy, mod, hard, mean]
                    best_threshold = threshold
                print("threshold {}: {} {} {} {}".format(threshold, easy, mod, hard, mean))
send_email(result[0], result[1], result[2], result[3], best_threshold)


                
            
    