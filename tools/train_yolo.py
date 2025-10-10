import sys
if "/ws/MambaIR" not in sys.path:
    # Add the project root to the system path
    # This allows importing modules from the MambaIR project
    # without needing to install it as a package.
    sys.path.append("/ws/MambaIR")

from yolov12.ultralytics.models.yolo import YOLO


data_cfg_path = '/ws/MambaIR/experiments/MambaIRv2_x2_xview/yolo_frozen_mamba.yaml'

yolo_path = '/ws/MambaIR/experiments/MambaIRv2_x2_xview/yolo_results/train7/weights/best.pt'


model = YOLO(yolo_path)

results = model.train(
    data=data_cfg_path,
    epochs=80,
    batch=180,
    lr0=0.001,
    imgsz=320,
    # scale=1.0,  # S:0.9; M:0.9; L:0.9; X:0.9
    # mosaic=0.5,
    # mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    # copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    optimizer='SGD',
    project='/ws/MambaIR/experiments/MambaIRv2_x2_xview/yolo_results',
    device=[0,1,2,3],
    patience=15,
    workers=24
)