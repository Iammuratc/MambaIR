from ultralytics import YOLO

def dummy_fuse(self, verbose=True):
    return self

model = YOLO('/home/louis/workspace/MambaIR/experiments/MambaIRv2_SR_x4/models/yolo_latest.pt')
# model.model.fuse = dummy_fuse.__get__(model, model.__class__)
# model = YOLO("yolov12n.pt")

# Evaluate model performance on the validation set
metrics = model.val(data="/home/louis/workspace/MambaIR/train_yolo/data-dota-4x.yaml")

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()