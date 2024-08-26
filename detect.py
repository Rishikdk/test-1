# from ultralytics import YOLO

# # Track objects in a video
# model = YOLO("D:/Project-6th/test-1/runs/detect/train5/weights/best.pt")
# model.track(source="D:/Project-6th/test-1/test-vedio.mp4 ", conf=0.4, save=True, show=True )

from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO("D:/Project-6th/test-1/runs/detect/train8/weights/best.pt")
results = model.predict(source="0", tracker="bytetrack.yaml", conf=0.4, save=True, show=True)