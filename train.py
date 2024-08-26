
from ultralytics import YOLO
if __name__ == "__main__":

     model = YOLO('yolov8m.pt')
     model.train(data="D:/Project-6th/test-1/dataset-3/data.yaml", epochs=30, imgsz=120)



                