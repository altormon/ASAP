## pip install ultralytics
## pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import os
import torch
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640, batch=8, single_cls=True, device="cuda" if torch.cuda.is_available() else "cpu")
