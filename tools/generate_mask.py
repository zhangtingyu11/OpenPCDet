from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import cv2
from math import *
from pathlib import Path
import logging
import pickle

ANGLE_LIMIT = 40.69
RANGE_LIMIT = 80
RANGE_NUM = 8

l_model_type = 'vit_l'
h_model_type = 'vit_h'
b_model_type = 'vit_b'

h_checkpoint_path = "./weights/sam_vit_h_4b8939.pth"
l_checkpoint_path = "./weights/sam_vit_l_0b3195.pth"
b_checkpoint_path = "./weights/sam_vit_b_01ec64.pth"

logging.getLogger().setLevel(logging.INFO)

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str

class Segment_Ground_Truth_KITTI:
    def __init__(self, data_root: str, split: str, choosen_class = None, device = 0) -> None:
        assert split in ['train', 'val', 'trainval']
        self.data_root = Path(data_root)
        self.split = split
        self.imagesets_path = self.data_root / 'ImageSets' / (split + '.txt')
        self.choosen_class = choosen_class
        self.sam = sam_model_registry[h_model_type](checkpoint=h_checkpoint_path)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            min_mask_region_area = 1000)
        
    def read_image(self, frame_id):
        frame_id_str = str(frame_id).zfill(6)
        image_name = frame_id_str+'.png'
        if self.split in ['train', 'val', 'trainval']:
            image_path = self.data_root / 'training' / 'image_2' /  image_name
        else:
            image_path = self.data_root / 'testing' / 'image_2' /  image_name
        self.image = cv2.imread(str(image_path))
    
    def generate_mask(self):
        masks = self.mask_generator.generate(self.image)
        return masks
    
    def plot_img(self):
        plt.figure(figsize=(20,20))
        plt.imshow(self.image)
    
    def show_anns(self, anns):
        plt.figure(figsize=(20,20))
        plt.imshow(self.image)

        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        img = (img*255).astype(np.uint8)
        img = cv2.addWeighted(self.image, 1-0.35, img[:, :, :3], 0.35, 0)
        return img
            
    def add_box(self, img, box):
        tlx, tly, brx, bry = box
        width, height = (brx-tlx), (bry-tly)
        centerx = (tlx+brx)/2
        centery = (tly+bry)/2

        # rect = patches.Rectangle((tlx, tly), width, height, linewidth=2, edgecolor='r', facecolor='none')  
        # self.ax.add_patch(rect)
        # self.ax.plot(centerx, centery, 'ro')
        img=cv2.rectangle(img, (floor(tlx), floor(tly)), (ceil(brx), ceil(bry)), (0, 255, 0), 2)
        cv2.circle(img, (round(centerx), round(centery)), radius = 4, color=(0, 0, 255), thickness=-1)
        return img

    def add_boxes(self, boxes):
        for box in boxes:
            self.add_box(box)
    
    def show(self):
        plt.axis('off')
        plt.show() 
        
    def save_img(self, img, frame_id, cls_type, cnt, data_root=None, split=None, save_to_database=True, angle_idx=None, range_idx=None):
        if split is None:
            split = self.split
        split = "image_gt_database_" + str(split) 
        
        if data_root is None:
            data_root = self.data_root
        image_name = [str(frame_id).zfill(6), cls_type, str(cnt)]
        save_dir = data_root / split
        if(not os.path.exists(save_dir.resolve().as_posix())):
            os.makedirs(save_dir)
        save_address = data_root / split / ('_'.join(image_name) + '.png')
        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        cv2.imwrite(save_address.resolve().as_posix(), result)
        if(save_to_database and angle_idx is not None and range_idx is not None):
            self.database[int(angle_idx)][int(range_idx)].append('_'.join(image_name) + '.png')
        
    def mat_save_img(self, filename='sam_mask'):
        plt.axis('off')
        plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    
    def custom_save_img(self, img, image_name, image_type = 'png', root = None, transparent = True):
        image_fullname = '.'.join([image_name, image_type])
        if root is not None:
            save_address = '/'.join([root, image_fullname])
        else:
            save_address = image_fullname
        # if transparent:
        #     result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # else:
        #     result = img
        cv2.imwrite(save_address, img)
        
        
    def save_single_scene(self, frame_id, masks):
        frame_id_str = str(frame_id).zfill(6)
        assert self.split in ['train', 'val', 'trainval']
        label_txt = self.data_root / 'training' / 'label_2' / (frame_id_str+'.txt')
        objects = get_objects_from_label(str(label_txt))
            
        object_centers = []
        img_height, img_width= masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1],
        for obj in objects:
            tlx, tly, brx, bry = obj.box2d
            center_y = (tlx+brx)/2
            center_x = (tly+bry)/2
            object_centers.append((center_x, center_y))
        cnt = 1
        masked_img = np.zeros((img_height, img_width), dtype = np.int32)
        for ann in masks:
            seg = ann['segmentation']
            masked_img[seg] = cnt
            cnt+=1
        
        for idx, (centerx, centery) in enumerate(object_centers):
            obj = objects[idx]
            if(obj.occlusion>0 or obj.cls_type not in self.choosen_class or obj.truncation > 0):
                continue
            final_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
            mask = (masked_img==masked_img[int(centerx)][int(centery)]).astype(np.uint8)
            mask *= 255
            mask.astype(np.uint8)
            k = np.ones((5, 5), np.uint8)
            close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=20)
            close = close.astype(np.bool_)
            final_mask[close, :3]= self.image[close]
            final_mask[close, 3] = 255
            label = objects[idx].box2d
            res, tlx, tly, brx, bry = self.judge_valid(final_mask, label)
            if(res):
                loc = objects[idx].loc
                x, y = loc[0], loc[2]
                degree = np.arctan2(y, x) * 180 / np.pi
                #* 从左往右序号分别是0, 1, 2 ...
                angle_idx = (degree-(90-self.angle_limit))//self.angle_interval
                if(angle_idx < 0 or angle_idx >= self.angle_num):
                    continue
                
                dis = np.sqrt(x*x + y*y)
                range_idx = dis//(self.range_limit/self.range_num)
                if(range_idx < 0 or range_idx >= self.range_num):
                    continue
                self.save_img(final_mask[tly:bry+1,tlx:brx+1], frame_id, objects[idx].cls_type, idx, angle_idx = angle_idx, range_idx = range_idx)
        logging.info('Frame id {} Finished'.format(frame_id_str))
    
    def generate_database_images(self):
        image_set_file = self.data_root / 'ImageSets' / (self.split + '.txt')
        with open(image_set_file, 'r') as f:
            lines = f.readlines()
        samples = list(map(int, lines))
        self.angle_limit = ANGLE_LIMIT
        self.range_limit = RANGE_LIMIT
        self.range_num = RANGE_NUM
        self.angle_num = int((ANGLE_LIMIT*2)/(360/64))
        self.angle_interval = 360/64
        self.database = [ [ [] for _ in range(self.range_num) ] for _ in range(self.angle_num) ]
        for sample in samples:
            self.read_image(sample)
            masks = self.generate_mask()
            self.save_single_scene(sample, masks)
        save_address = self.data_root / 'image_database_train.pkl'
        with open(save_address.resolve().as_posix(), 'wb') as f:
            pickle.dump(self.database, f)
            
    def judge_valid(self, image, label, iou = 0.7):
        conv_image = (image[:, :, 3:4]>0).astype(np.uint8)
        conv_image*=255
        contours, _ = cv2.findContours(conv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
        max_area = 0
        max_rect = None

        for rect in bounding_rects:
            x, y, w, h = rect
            area = w * h

            if area > max_area:
                max_area = area
                max_rect = rect
        x, y, w, h = max_rect
        tlx, tly, brx, bry = x, y, x+w, y+h
        qbox_area = (brx - tlx) * (bry - tly)
        iw = (min(brx, label[2]) -
                max(tlx, label[0]))
        ua = 0
        if iw > 0:
            ih = (min(bry, label[3]) -
                    max(tly, label[1]))
            if ih > 0:
                ua = (
                    (label[2] - label[0]) *
                    (label[3] - label[1]) + qbox_area - iw * ih)    
        return (iw * ih / ua) >= iou, tlx, tly, brx, bry
    
    
    def show_mask_with_black_and_white(self, box, masks):
        tlx, tly, brx, bry = box
        centery = (tlx+brx)/2
        centerx = (tly+bry)/2
        img_height, img_width= masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1],
        cnt = 1
        masked_img = np.zeros((img_height, img_width), dtype = np.int32)
        for ann in masks:
            seg = ann['segmentation']
            masked_img[seg] = cnt
            cnt+=1
        #! 生成初始的黑白的mask
        mask = (masked_img==masked_img[int(centerx)][int(centery)]).astype(np.uint8)
        black_white_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        black_white_mask *= 255
        plt.imshow(black_white_mask)
        self.custom_save_img(black_white_mask, 'black_white_original_mask', transparent=False)
        
        #! 生成初始的mask后的透明图片
        # final_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        # mask = (masked_img==masked_img[int(centerx)][int(centery)])
        # final_mask[mask, :3]= self.image[mask]
        # final_mask[mask, 3] = 255
        # plt.imshow(final_mask)
        # self.custom_save_img(final_mask, 'original_mask')

        
        #! 经过闭运算后的黑白mask
        mask = (masked_img==masked_img[int(centerx)][int(centery)]).astype(np.uint8)
        mask *= 255
        mask.astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=20)
        black_white_close = np.repeat(close[:, :, np.newaxis], 3, axis=2)
        plt.imshow(black_white_close)
        self.custom_save_img(black_white_close, 'black_white_close_mask', transparent=False)
        
        # #* 显示label和轮廓的包围框
        left, right, top, bottom = img_width-1, 0, img_height-1, 0
        for x in range(img_width):
            for y in range(img_height):
                if(black_white_close[y][x][0] > 0):
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
                    
        cv2.rectangle(black_white_close, 
                      (floor(tlx), floor(tly)), (ceil(brx), ceil(bry)),
                      (0, 255, 0), 
                      2)
        cv2.rectangle(black_white_close, 
                (floor(left), floor(top)), (ceil(right), ceil(bottom)),
                (0, 0, 255), 
                2)
        self.custom_save_img(black_white_close, 'gt_and_pred', transparent=False)
        
        #* 计算iou
        tlx, tly, brx, bry = left, top, right, bottom
        qbox_area = (brx - tlx) * (bry - tly)
        iw = (min(brx, box[2]) -
                max(tlx, box[0]))
        ua = 0
        ih = 0
        if iw > 0:
            ih = (min(bry, box[3]) -
                    max(tly, box[1]))
            if ih > 0:
                ua = (
                    (box[2] - box[0]) *
                    (box[3] - box[1]) + qbox_area - iw * ih)    
        print("iou = {}".format(iw * ih / ua))
        
        
        #! 闭运算后的mask后的透明图片
        final_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        mask = (masked_img==masked_img[int(centerx)][int(centery)]).astype(np.uint8)
        mask *= 255
        mask.astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=20)
        close = close.astype(np.bool_)
        final_mask[close, :3]= self.image[close]
        final_mask[close, 3] = 255
        plt.imshow(final_mask)
        self.custom_save_img(final_mask, 'close_mask')
        
if __name__ == '__main__':
    choosen_classes = ['Car', 'Pedestrian', 'Cyclist']
    sgtk = Segment_Ground_Truth_KITTI('./data/kitti', 'train', choosen_classes)
    #! 34
    # sgtk.read_image(34)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((46.17, 196.15, 328.40, 286.09), masks)
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (46.17, 196.15, 328.40, 286.09))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    #! 3
    # sgtk.read_image(3)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((614.24, 181.78, 727.31, 284.77), masks)
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (614.24, 181.78, 727.31, 284.77))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    #! 29
    # sgtk.read_image(29)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((652.31, 174.94, 690.16, 204.97), masks)
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (652.31, 174.94, 690.16, 204.97))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    # sgtk.show()
    
    sgtk.generate_database_images()

