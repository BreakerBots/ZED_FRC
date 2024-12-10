# BreakerBots - ZED Depth Camera Object Detection For FRC

This is the co-processor side backend program to control a StereoLabs ZED depth camera for true 3D object detection with YOLO object detection. 

### Features
 - Optional visualization tools 
    - The camera point cloud is displayed in a 3D OpenGL view
    - 3D bounding boxes around detected objects are drawn
 - Easy configuration through JSON file
    - Set inference parameters, NN weight filepaths, camera video settings, depth calculation settings, and more
    - Ability to set per-class filtering parameters
 - End-to-end CUDA acceleration
 - Integrated NT publishing for FRC
    - Robot API available [HERE]() (WIP)

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
## Setting up (WIP)

 - Install yolov8 using pip

```sh
pip install ultralytics
```

## Run the program

*NOTE: The ZED v1 is not compatible with this program*

```
python detector.py --settings settings.json # [--svo path/to/file.svo]
```

## Training your own model

This program can use any model trained with YOLOv8, including custom trained one. For a getting started on how to trained a model on a custom dataset with YOLOv5, see here https://docs.ultralytics.com/tutorials/train-custom-datasets/