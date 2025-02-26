from ultralytics import YOLO
import torch

# load a  pretrained model (recommended for training)
model = YOLO('yolov8m-seg.pt')

# empty cuda cache, avoid out of memory
torch.cuda.empty_cache()

# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640, batch=-1)
