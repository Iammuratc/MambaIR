import sys
if "/ws/MambaIR" not in sys.path:
    # Add the project root to the system path
    # This allows importing modules from the MambaIR project
    # without needing to install it as a package.
    sys.path.append("/ws/MambaIR")

from yolov12.ultralytics.models.yolo import YOLO


data_cfg_path = '/ws/MambaIR/experiments/yolo_xview_gsd_30/yolo_train.yaml'

yolo_path = '/ws/MambaIR/runs/detect/train/weights/best.pt'


model = YOLO(yolo_path)



# This runs validation and returns metrics
metrics = model.val(
    data='/ws/MambaIR/experiments/yolo_xview_gsd_30/yolo_train.yaml',  # Your data config
    project='/ws/MambaIR/experiments/yolo_xview_gsd_30/test_results',  # Project to save results
    split='test',  # or 'test'
    batch=80,
    imgsz=320,
    iou = 0.60,  # IoU threshold for NMS
    save_json=True,  # Save results to JSON for analysis
    save_hybrid=True,  # Save labels with additional info
    device=[0,1,2,3]
)

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")