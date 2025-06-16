from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/zty/Project/DeepLearning/OpenPCDet/data/kitti_coco/kitti_coco.yaml", epochs=100, imgsz=640,
                      classes=[0,2,3,4,5,7,8], cos_lr=True, dfl=0,
                      lr0=0.001, single_cls=True)