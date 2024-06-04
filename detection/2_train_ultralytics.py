import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLOv10

if __name__ == '__main__':
    # Load a model
    model = YOLOv10('yolov10x.pt')  # build a new model from YAML

    # Train the model
    results = model.train(data='./detection/ee443.yaml', epochs=200, imgsz=640)