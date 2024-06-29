# ZED SDK - Object Detection

This sample shows how to detect custom objects using the official Pytorch implementation of YOLOv8 from a ZED camera and ingest them into the ZED SDK to extract 3D informations and tracking for each objects.

### Features
 - Optional visualization tools 
    - The camera point cloud is displayed in a 3D OpenGL view
    - 3D bounding boxes around detected objects are drawn
 - Easy configuration through JSON file
    - Set inference parameters, NN weight filepaths, camera video settings, depth calculation settings, and more
 - Integraated NT publishing for FRC
    - Robot API available [HERE]() (WIP)

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
## Setting up (WIP)

 - Install yolov8 using pip

```sh
pip install ultralytics
```

## Run the program

*NOTE: The ZED v1 is not compatible with this module*

```
python detector.py --settings settings.json # [--svo path/to/file.svo]
```

## Training your own model

This program can use any model trained with YOLOv8, including custom trained one. For a getting started on how to trained a model on a custom dataset with YOLOv5, see here https://docs.ultralytics.com/tutorials/train-custom-datasets/

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/