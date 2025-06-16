from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    # 计算交集的坐标
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    # 计算交集的面积
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    # 计算并集的面积
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2b - x1b + 1) * (y2b - y1b + 1)
    union_area = box1_area + box2_area - inter_area
    # 计算IOU
    iou = inter_area / union_area
    return iou

if __name__ == "__main__":
    model_path = "runs/detect/train6/weights/best.pt"
    model = YOLO(model_path)
    
    total_gt_boxes = 0
    detected_gt = 0
    image_files = list(Path("/home/zty/Project/DeepLearning/OpenPCDet/data/kitti_coco/images/val/").glob("*.png"))
    for image_file in tqdm(image_files):
        result = model.predict(image_file, conf=0.01, verbose=False)
        boxes = result[0].boxes
        cls = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        orig_shape = boxes.orig_shape
        height, width = orig_shape
        
        label_filename = str(image_file).replace("images", "labels").replace(".png", ".txt")
        gt_boxes = []
        with open(label_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_index, x, y, w, h = line.split()
                class_index = int(class_index)
                x, y, w, h = float(x), float(y), float(w), float(h)
                x, y, w, h = x * width, y * height, w * width, h * height
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                if class_index not in [0, 1, 3, 4, 5, 7, 8]:
                    continue
                gt_boxes.append([x1, y1, x2, y2])
        total_gt_boxes += len(gt_boxes)
        for gt_box in gt_boxes:
            for pred_box in xyxy:
                if iou(gt_box, pred_box) > 0.5:
                    detected_gt += 1
                    break
    recall = detected_gt / total_gt_boxes
    print(recall)
                