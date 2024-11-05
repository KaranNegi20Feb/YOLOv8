from ultralytics import YOLO
import torch

# Check if CUDA is available and set the device accordingly
device = 0 if torch.cuda.is_available() else 'cpu'

# Load the YOLO model
model = YOLO('human.pt')  # Ensure 'human.pt' is the correct model path

# Run detection on the webcam feed
results = model.predict(source=0, show=True, conf=0.6, save=True, device=device)


