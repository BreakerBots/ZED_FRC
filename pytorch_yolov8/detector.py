#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import ntcore as nt
import json

lock = Lock()
run_signal = False
exit_signal = False

ntInst = nt.NetworkTableInstance.getDefault()
ntInst.startClient4("ZED")
ntInst.setServerTeam(5104)
ntInst.startDSClient()
table = ntInst.getTable("ZED_detections")
heartbeatPub = table.getIntegerTopic("heartbeat").publish()
idPub = table.getIntegerArrayTopic("id").publish()
labelPub = table.getStringArrayTopic("label").publish()
latencyPub = table.getIntegerTopic("pipeline_latency").publish()
xVelPub = table.getDoubleArrayTopic("x_vel").publish()
yVelPub = table.getDoubleArrayTopic("y_vel").publish() 
zVelPub = table.getDoubleArrayTopic("z_vel").publish()
xPub = table.getDoubleArrayTopic("x").publish()
yPub = table.getDoubleArrayTopic("y").publish() 
zPub = table.getDoubleArrayTopic("z").publish()
boxLenPub = table.getDoubleArrayTopic("box_l").publish()
boxWidthPub = table.getDoubleArrayTopic("box_w").publish() 
boxHeightPub = table.getDoubleArrayTopic("box_h").publish() 
confPub = table.getDoubleArrayTopic("conf").publish() 
isVisPub = table.getBooleanArrayTopic("is_visible").publish()
isMovingPub = table.getBooleanArrayTopic("is_moving").publish()
ntHeartbeat = 0


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres, iou_thres):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    torch.cuda.set_device(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(weights, task='detect')
    model.to(device=device)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections

    print("Loading Settings...")
    settingsFile = open(opt.settings)
    settings = json.load(settingsFile)



    capture_thread = Thread(target=torch_thread, kwargs={'weights': settings["inference"]["weights"], 'img_size': settings["inference"]["size"], "conf_thres": settings["inference"]["conf_thresh"],  "iou_thres": settings["inference"]["iou_thresh"]})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    visualize = settings["visualize"]
    publish = settings["networktables"]["publish"]

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = depthModeFromString(settings["depth"]["mode"])
    init_params.camera_resolution = resolutionFromString(settings["camera"]["resolution"])
    init_params.camera_fps = settings["camera"]["fps"]
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = settings["depth"]["max_dist"]
    init_params.depth_minimum_distance = settings["depth"]["min_dist"]
    init_params.sdk_verbose = 1

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = settings['depth']["depth_conf_thresh"]
    runtime_params.texture_confidence_threshold = settings['depth']["texture_conf_thresh"]
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()
    setCameraVideoSettings(zed, settings)

    print("Initialized Camera")  

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer = None
    if (visualize):
        viewer = gl.GLViewer()
        viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = None
    if (visualize):
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()
    try:
        while (not visualize or viewer.is_available()) and not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # -- Get the image
                lock.acquire()
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                lock.release()
                run_signal = True

                # -- Detection running on the other thread
                while run_signal:
                    sleep(0.001)

                # Wait for detections
                lock.acquire()
                # -- Ingest detections
                zed.ingest_custom_box_objects(detections)
                lock.release()
                zed.retrieve_objects(objects, obj_runtime_param)

                if (publish):
                    publishNT(zed, objects)

                if (visualize):
                    # -- Display
                    # Retrieve display data
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
                    point_cloud.copy_to(point_cloud_render)
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                    # 3D rendering
                    viewer.updateData(point_cloud_render, objects)
                    # 2D rendering
                    np.copyto(image_left_ocv, image_left.get_data())
                    cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
                    global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                    # Tracking view
                    track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

                    cv2.imshow("ZED | 2D View and Birds View", global_image)
            
                key = cv2.waitKey(10)
                if key == 27:
                    exit_signal = True
            else:
                exit_signal = True

        if (visualize):
            viewer.exit()
        exit_signal = True
        zed.close()
    except KeyboardInterrupt:
        if (visualize):
            viewer.exit()
        exit_signal = True
        zed.close()

def depthModeFromString(string):
    string = string.upper()
    if (string == "NEURAL"):
        return sl.DEPTH_MODE.NEURAL
    elif (string == "NEURAL_PLUS"):
        return sl.DEPTH_MODE.NEURAL_PLUS
    elif (string == "PERFORMANCE"):
        return sl.DEPTH_MODE.PERFORMANCE
    elif (string == "QUALITY"):
        return sl.DEPTH_MODE.QUALITY
    else:
        return sl.DEPTH_MODE.ULTRA
    
def resolutionFromString(string):
    string = string.upper()
    if (string == "HD2K"):
        return sl.RESOLUTION.HD2K
    elif (string == "HD1080"):
        return sl.RESOLUTION.HD1080
    elif (string == "HD1200"):
        return sl.RESOLUTION.HD1200
    elif (string == "HD720"):
        return sl.RESOLUTION.HD720
    elif (string == "SVGA"):
        return sl.RESOLUTION.SVGA
    elif (string == "VGA"):
        return sl.RESOLUTION.VGA
    else:
        return sl.RESOLUTION.AUTO

def setCameraVideoSettings(camera, settings):
    camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, settings["camera"]["brightness"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, settings["camera"]["contrast"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.HUE, settings["camera"]["hue"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, settings["camera"]["saturation"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, settings["camera"]["sharpness"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, settings["camera"]["gamma"])
    exp = settings["camera"]["exposure"]
    gain = settings["camera"]["gain"]
    if (exp < 0 or gain < 0):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
    else:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, settings["camera"]["exposure"])
        camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, settings["camera"]["gain"])
    wb = settings["camera"]['wb']
    if (wb < 2800 or wb > 6500):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
    else:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)
    return

def publishNT(camera, objects):
    global ntHeartbeat
    xArr = []
    yArr = []
    zArr = []
    xVelArr = []
    yVelArr = []
    zVelArr = []
    idArr = []
    labelArr = []
    bLenArr = []
    bHeightArr = []
    bWidthArr = []
    confArr = []
    isVisArr = []
    isMovArr = []

    heartbeatPub.set(ntHeartbeat)
    ntHeartbeat+=1
    latencyPub.set(camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds() - objects.timestamp.get_nanoseconds())

    objList = objects.object_list
    numObjs = __builtins__.len(objList)
    for obj in objList:
        idArr.append(obj.id)
        confArr.append(obj.confidence)
        isVisArr.append(obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
        isMovArr.append(obj.action_state == sl.OBJECT_ACTION_STATE.MOVING)
        # labelArr[x] = obj.
        pos = obj.position
        xArr.append(pos[0])
        yArr.append(pos[1])
        zArr.append(pos[2])
        vel = obj.velocity
        xVelArr.append(vel[0])
        yVelArr.append(vel[1])
        zVelArr.append(vel[2])
        dims = obj.dimensions
        bWidthArr.append(dims[0])
        bHeightArr.append(dims[1])
        bLenArr.append(dims[2])

    
   
    xPub.set(xArr)
    yPub.set(yArr)
    zPub.set(zArr)

        
        



    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default="settings.json", help='settings.json path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
