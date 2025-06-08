from ultralytics import YOLO
import torch.serialization

# Allowlist the DetectionModel class
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

model = YOLO("yolov8n.pt")  # Download pretrained model
model.export(format="onnx", imgsz=640, optimize=False)