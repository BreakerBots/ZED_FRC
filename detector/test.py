from ultralytics import YOLO

model = YOLO("models/2025/best.pt")

# Export the model
model.export(format="onnx")
