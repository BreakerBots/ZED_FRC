from ultralytics import YOLO

model = YOLO("models/xqJ46o_100.pt")

# Export the model
model.export(format="onnx")