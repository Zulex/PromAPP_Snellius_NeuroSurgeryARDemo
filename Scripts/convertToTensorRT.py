from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("Models/Keypose_90kImages_run18_best.pt")

# Export the model to TensorRT format
model.export(format="engine", device = 0)  # creates 'yolov8n.engine'
