from ultralytics import YOLO

# Use a small model first for speed
model = YOLO("yolov8s.pt")  # or yolov8n.pt for even faster, smaller

# Train
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU
)

