from ultralytics import YOLO

# Load your YOLOv8 PyTorch model (.pt)
model = YOLO("yolov8_model.pt")   # replace with your trained .pt path

# Export directly to TFLite
model.export(format="tflite", dynamic=False, int8=False)

# This will generate: yolov8n.tflite in the same directory
print("✅ Export complete! Check the generated .tflite file.")
