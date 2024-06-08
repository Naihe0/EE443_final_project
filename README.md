# EE 443 2024 Challenge: Single Camera Multi-Object Tracking

### TA: Chris Yang (cycyang), Chris Yin (c8yin)

### Team Name: You Are Right

### Team Members: Tony Gu, Ningrui Yang, Haoxiong Zhang

### Task Description
The EE 443 2024 Challenge: Single Camera Multi-Object Tracking aims to enhance the performance of object detection and tracking algorithms in single-camera environments. Participants will focus on improving detection models, ReID (Re-identification) models, and Multi-Object Tracking (MOT) algorithms to achieve superior object tracking accuracy.

### Baseline Code for Detection

1. Install ultralytics (follow the [Quickstart - Ultralytics](https://docs.ultralytics.com/quickstart/#install-ultralytics))

2. Download the `data.zip` from GDrive link provided in the Ed Discussion

The folder structure should look like this:
```
├── data
│   ├── test
│   ├── train
│   └── val
├── detection
│   ├── 1_prepare_data_in_ultralytics_format.py
│   ├── 2_train_ultralytics.py
│   ├── 3_inference_ultralytics.py
│   └── ee443.yaml
```

4. Prepare the dataset into ultralytics format (remember to modified the path in the script)
```
python3 detection/1_prepare_data_in_ultralytics_format.py
```
After the script, the `ultralytics_data` folder should looke like this:
```
├── data
├── detection
├── ultralytics_data
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
```

4. Train the model using ultralytics formatted data (remember to modified the path in the script and config file `ee443.yaml`)
```
python3 detection/2_train_ultralytics.py
```
The model will be saved to `runs/detect/` with an unique naming.

5. Inference the model using the testing data (remember to modify the path in the script)
```
python3 detection/3_inference_ultralytics.py
```

### Baseline Code for Tracking

6. Extract the features
```
python3 reid/3_inference_ultralytics.py
```

7. Tracking
```
python3 tracking/main.py
```

### StrongSORT Code for Tracking

8. Tracking (only compatible with YOLOv7 model)
```
cd strong_sort_work_dir/sort
python3 track_v7.py --yolo-weight [dir_to/detection_model].pt --strong-sort-weights weights/osnet_x0_25_msmt17 --line-thickness 2 --classes 0 2 7 --save-txt --count --source [dir_to/raw_video_to_track].avi
```

8.1 Produce the raw video for tracking (remember to modify the path to the raw images)
```
python3 frame.py
```

8.2 Train a YOLOv7 detection model (remember to modify the path in test.yaml)
```
cd strong_sort_work_dir/yolov7
python3 train.py --weights yolov7_training.pt --data data/test.yaml --cfg cfg/training/yolov7.yaml --epochs 150 --batch-size 16
```

8.3 Reformat the .txt to "[camera_id],[track_id],[frame_id],[xmin],[ymin],[width],[height],[-1],[-1]". (remember to change the camera_id and dir paths in the script)
```
python3 reformat_txt.py
```
