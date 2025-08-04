from ultralytics import YOLO


model = YOLO('/home/louis/workspace/MambaIR/experiments/MambaIRv2_SR_x4/models/yolo_latest.pt').model

# Evaluate model performance on the validation set
metrics = model.val(data="/home/louis/workspace/MambaIR/train_yolo/data-dota-4x.yaml")

# Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()