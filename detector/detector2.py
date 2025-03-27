#!/usr/bin/env python3

import sys
import threading
import numpy as np

import argparse

import yaml
# import torch
import cv2
import pyzed.sl as sl

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import ntcore as nt
from wpimath import geometry as geom
from enum import Enum
from flask import Flask, Response
from werkzeug.serving import make_server

# Create Flask app
app = Flask(__name__)

lock = Lock()
run_signal = False
exit_signal = False
ntHeartbeat = 0
global_image = np.zeros((720, 1280, 3), dtype=np.uint8) 

class CameraType(Enum):
    ZED_X = 1
    ZED_2 = 2

class ServerThread(threading.Thread):

    def __init__(self, host, port):
        threading.Thread.__init__(self)
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

def start_server():
    global server
    # App routes defined here
    server = ServerThread(app)
    server.start()

def stop_server():
    global server
    server.shutdown()

# Initialize the Flask route for streaming
@app.route('/')
def video_feed():
    def generate():
        while not exit_signal:
            # Capture the image and encode it in JPEG format
            # lock.acquire()
            ret, jpeg = cv2.imencode('.jpg', global_image)
            # lock.release()
            if ret:
                # Return the image as a response with the correct MIME type for MJPEG
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            sleep(0.01)  # Avoid 100% CPU usage
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def configNT(settings):
    global heartbeatPub, idPub, labelPub, latencyPub, transPub, boxPub, confPub, isVisPub, isMovingPub, camPosePub, camOriginPub, camPoseLatencyPub, camPoseConfPub, ntInst
    ntInst = nt.NetworkTableInstance.getDefault()
    ntInst.startClient4(settings["networktables"]["name"])
    ntInst.setServerTeam(settings["networktables"]["team"])
    ntInst.startDSClient()
    mainTable = ntInst.getTable(settings["networktables"]["name"])

    table = mainTable.getSubTable("detections")
    heartbeatPub = table.getIntegerTopic("heartbeat").publish()
    idPub = table.getIntegerArrayTopic("id").publish()
    labelPub = table.getStringArrayTopic("label").publish()
    latencyPub = table.getIntegerTopic("pipeline_latency").publish()
    # xVelPub = table.getDoubleArrayTopic("x_vel").publish()
    # yVelPub = table.getDoubleArrayTopic("y_vel").publish() 
    # zVelPub = table.getDoubleArrayTopic("z_vel").publish()
    transPub = table.getStructArrayTopic("translation", geom.Translation3d).publish()
    boxPub = table.getStructArrayTopic("box", geom.Translation3d).publish()
    confPub = table.getDoubleArrayTopic("conf").publish() 
    isVisPub = table.getBooleanArrayTopic("is_visible").publish()
    isMovingPub = table.getBooleanArrayTopic("is_moving").publish()
    camPoseTable = mainTable.getSubTable("pose")
    camPosePub = camPoseTable.getStructTopic("cam_pose", geom.Pose3d).publish()
    camOriginPub = camPoseTable.getStructTopic("cam_pose_origin", geom.Pose3d).publish()
    camPoseLatencyPub = camPoseTable.getDoubleTopic("cam_pose_latency").publish()
    camPoseConfPub = camPoseTable.getDoubleTopic("cam_pose_conf").publish()

def main():
    global exit_signal, global_image, flask_thread, fps
    fps = 0.0

    print("Loading Settings...")
    settingsFile = open(opt.settings)
    settings = yaml.full_load(settingsFile)



    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    viz_ogl = settings["visualization"]["desktop"]["point_cloud_3d"]
    viz_ocv_disp = settings["visualization"]["desktop"]["ocv_2d"]
    viz_webserver = settings["visualization"]["webserver"]["enable"]
    viz_ocv_backend = viz_ocv_disp or viz_webserver
    
    if (viz_webserver):
         # Start Flask app in a separate thread
        flask_thread = ServerThread(settings["visualization"]["webserver"]["host"], settings["visualization"]["webserver"]["port"])
        flask_thread.start()
    

    publish = settings["networktables"]["publish"]
    if publish:
        configNT(settings)

    cameraSettings = yaml.full_load(open(settings["general"]["camera_config"]))
    cameraType = cameraTypeFromString(cameraSettings["model"])

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = depthModeFromString(settings["depth"]["mode"])
    init_params.camera_resolution = resolutionFromString(cameraSettings["resolution"], cameraType)
    init_params.camera_fps = cameraSettings["fps"]
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP #sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = settings["depth"]["max_dist"]
    init_params.depth_minimum_distance = settings["depth"]["min_dist"]
    init_params.depth_stabilization = settings["depth"]["stabilization"]
    init_params.enable_image_enhancement = cameraSettings["image_enhancement"]
    init_params.camera_disable_self_calib = not cameraSettings["camera_self_calib"]
    init_params.sdk_verbose = 1

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = settings['depth']["depth_conf_thresh"]
    runtime_params.texture_confidence_threshold = settings['depth']["texture_conf_thresh"]
    runtime_params.remove_saturated_areas = settings['depth']["remove_saturated_areas"]
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()
    setCameraVideoSettings(cameraType, zed, cameraSettings)

    print("Initialized Camera")  

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    positional_tracking_parameters.enable_area_memory = settings["localization"]["area_memory"]
    positional_tracking_parameters.enable_pose_smoothing = settings["localization"]["pose_smoothing"]
    positional_tracking_parameters.set_gravity_as_origin = settings["localization"]["use_gravity_origin"]
    positional_tracking_parameters.depth_min_range = settings["localization"]["min_depth"]
    positional_tracking_parameters.mode = positionalTrackingModeFromString(settings["localization"]["mode"])
    zed.enable_positional_tracking(positional_tracking_parameters)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    objPeram, rtPerams, classes = startObjectDetectionFromYaml(settings["general"]["inference_config"], zed)


    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer

    # output_path = "svorecord.svo2"
    # recordingParameters = sl.RecordingParameters()
    # recordingParameters.compression_mode = sl.SVO_COMPRESSION_MODE.H264
    # recordingParameters.video_filename = output_path
    # err = zed.enable_recording(recordingParameters)
    
   
    viewer = None
    if (viz_ogl):
        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        point_cloud_render = sl.Mat()
        viewer = gl.GLViewer()
        viewer.init(camera_infos.camera_model, point_cloud_res, objPeram.enable_tracking)
        point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        
    if (viz_ocv_backend):
        # Utilities for 2D display
        image_left = sl.Mat()
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

        # Utilities for tracks view
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = None
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()
    try:
        while ((not viz_ocv_backend) or (not viz_ogl) or viewer.is_available()) and not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
               
                zed.retrieve_custom_objects(objects, rtPerams, 0)

                if (viz_ocv_disp or publish):
                    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)


                if (publish):
                    publishNT(zed, cam_w_pose, objects, classes)

                if (viz_ogl):
                    # -- Display
                    # Retrieve display data
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
                    point_cloud.copy_to(point_cloud_render)

                    # 3D rendering
                    viewer.updateData(point_cloud_render, objects)   

                    
                if (viz_ocv_backend):
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                    # 2D rendering
                    np.copyto(image_left_ocv, image_left.get_data())
                    cv_viewer.render_2D(image_left_ocv, image_scale, objects, objPeram.enable_tracking, classes)
    
                    global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                    cv2.putText(global_image, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
                    # Tracking view
                    track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

                    if (viz_ocv_disp):
                        cv2.imshow("ZED | 2D View and Birds View", global_image)
                    

                fps = zed.get_current_fps()
                print("fps: " + str(int(fps)))
                
            
                key = cv2.waitKey(10)
                if key == 27:
                    exit_signal = True
            else:
                exit_signal = True

        if (viz_ogl):
            viewer.exit()
        if (viz_webserver):
            flask_thread.shutdown()
        exit_signal = True
        zed.close()
        # zed.disable_recording()
    except KeyboardInterrupt:
        if (viz_ogl):
            viewer.exit()
        if (viz_webserver):
            flask_thread.shutdown()
        exit_signal = True
        zed.close()
        # zed.disable_recording()

def defaultIfNotValue(value, returnDefaultCheckLambda, default):
    if (returnDefaultCheckLambda(value)):
        return default
    else:
        return value

def startObjectDetectionFromYaml(inferenceSettingsPath, zed):
    inferenceConfig = yaml.full_load(open(inferenceSettingsPath))
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    obj_param.custom_onnx_file = inferenceConfig["weights"]
    obj_param.custom_onnx_dynamic_input_shape = sl.Resolution(inferenceConfig["input_shape"]["horizontal"],inferenceConfig["input_shape"]["vertical"])
    obj_param.enable_tracking = True
    obj_param.filtering_mode = filteringModeFromString(inferenceConfig["filter_type"])
    obj_param.prediction_timeout_s = defaultIfNotValue(inferenceConfig["global_defaults"]["tracking_timeout"], lambda x : x < 0, 0.2)
    obj_param.max_range = defaultIfNotValue(inferenceConfig["global_defaults"]["max_tracking_dist"], lambda x : x == 0 or x < 0, 10)
    zed.enable_object_detection(obj_param)
    
    defaultDetConf = inferenceConfig["global_defaults"]["conf_thresh"] * 100
    defaultProps = sl.CustomObjectDetectionProperties()
    defaultProps.detection_confidence_threshold = defaultDetConf
    defaultProps.tracking_timeout = obj_param.prediction_timeout_s
    defaultProps.tracking_max_dist =  obj_param.max_range

    classes = inferenceConfig["classes"]
    # props = []
    props = {}
    for x in range(0,len(classes)):
        # props.append(customObjectDetectionPropertiesFromConfig(classes[x], defaultProps))
        props[x] = customObjectDetectionPropertiesFromConfig(classes[x], defaultProps)

    rtPerams = sl.CustomObjectDetectionRuntimeParameters(defaultProps, props)

    # rtPerams.object_detection_properties = defaultProps
    # rtPerams.object_class_detection_properties = props

    zed.enable_object_detection(obj_param)

    return (obj_param, rtPerams, classes)

def customObjectDetectionPropertiesFromConfig(config, defaultProps):
    props = sl.CustomObjectDetectionProperties()
    props.enabled = config["enabled"]
    props.detection_confidence_threshold = defaultIfNotValue(config["conf_thresh"] * 100, lambda x : x < 0, defaultProps.detection_confidence_threshold)
    props.is_grounded = config["is_grounded"]
    props.is_static = config["is_static"]
    props.tracking_timeout = defaultIfNotValue(config["tracking_timeout"], lambda x : x < 0, defaultProps.tracking_timeout)
    props.tracking_max_dist =  defaultIfNotValue(config["max_tracking_dist"], lambda x : x == 0 or x < 0, defaultProps.tracking_max_dist)
    props.max_box_width_normalized = config["max_box_norm"]["width"]
    props.max_box_height_normalized  = config["max_box_norm"]["height"]
    props.min_box_width_normalized = config["min_box_norm"]["width"]
    props.min_box_height_normalized  = config["min_box_norm"]["height"]
    return props


    
def filteringModeFromString(string):
    string = string.upper()
    if (string == "NMS3D"):
        return sl.OBJECT_FILTERING_MODE.NMS3D
    elif (string == "NMS3D_PER_CLASS"):
        return sl.OBJECT_FILTERING_MODE.NMS3D_PER_CLASS
    else:
        return sl.OBJECT_FILTERING_MODE.NONE


def slPoseToWPILib(slPose):
    rotVec = slPose.get_rotation_vector()
    slTrans = sl.Translation()
    slPose.get_translation(slTrans)
    
    wpiRot = geom.Rotation3d(rotVec)
    wpiTrans = geom.Translation3d(slTrans.get())
    return geom.Pose3d(wpiTrans, wpiRot)


def positionalTrackingModeFromString(string):
    string = string.upper()
    if (string == "GEN_2" or string == "GEN-2" or string == "GEN2"):
        return sl.POSITIONAL_TRACKING_MODE.GEN_2
    else:
        return sl.POSITIONAL_TRACKING_MODE.GEN_1

def cameraTypeFromString(string):
    string = string.upper()  
    if (string == "ZEDX" or string == "ZEDXM"):
        return CameraType.ZED_X
    else:
        return CameraType.ZED_2
    

def colorSpaceConversionFromString(string):
    string = string.upper()
    if (string == "BGR"):
        return cv2.COLOR_BGRA2BGR
    elif (string == "RGBA"):
        return cv2.COLOR_BGRA2RGBA
    elif (string == "GRAY"):
        return cv2.COLOR_BGRA2GRAY
    elif (string == "BGR565"):
        return cv2.COLOR_BGRA2BGR565
    elif (string == "BGR555"):
        return cv2.COLOR_BGRA2BGR555
    elif (string == "YUV_I420"):
        return cv2.COLOR_BGRA2YUV_I420
    elif (string == "YUV_IYUV"):
        return cv2.COLOR_BGRA2YUV_IYUV
    elif (string == "YUV_YV12"):
        return cv2.COLOR_BGRA2YUV_YV12
    else:
        return cv2.COLOR_BGRA2RGB
    
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

def resolutionFromString(string, cameraType):
    string = string.upper()
    if (cameraType == CameraType.ZED_2):
        if (string == "HD2K"):
            return sl.RESOLUTION.HD2K
        elif (string == "HD1080"):
            return sl.RESOLUTION.HD1080
        elif (string == "HD720"):
            return sl.RESOLUTION.HD720
        elif (string == "VGA"):
            return sl.RESOLUTION.VGA
        else:
            return sl.RESOLUTION.AUTO
    else:
        if (string == "HD1200"):
            return sl.RESOLUTION.HD1200
        elif (string == "HD1080"):
            return sl.RESOLUTION.HD1080
        elif (string == "SVGA"):
            return sl.RESOLUTION.SVGA
        else:
            return sl.RESOLUTION.AUTO
        
def setCameraVideoSettings(cameraType, camera, settings):
    if (cameraType == CameraType.ZED_X):
        setCameraVideoSettingsZEDX(camera=camera, settings=settings)
    else:
        setCameraVideoSettingsZED2(camera=camera, settings=settings)
        
def setCameraVideoSettingsZED2(camera, settings):
    camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, settings["brightness"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, settings["contrast"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.HUE, settings["hue"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, settings["saturation"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, settings["sharpness"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, settings["gamma"])
    exp = settings["exposure"]
    gain = settings["gain"]
    if (exp < 0 or gain < 0):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
    else:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, settings["exposure"])
        camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, settings["gain"])
    wb = settings['wb']
    if (wb < 2800 or wb > 6500):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
    else:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)
    return

def setCameraVideoSettingsZEDX(camera, settings):
    camera.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, settings["saturation"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, settings["sharpness"])
    camera.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, settings["gamma"])
    exp = settings["exposure_time"]
    gain = settings["gain"]
    aGain = settings["analog_gain"]
    dGain = settings["digital_gain"]
    if (exp < 0 or gain < 0):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
    else:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_TIME, exp)
        camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, settings["gain"])

    if (aGain < 0 or dGain < 0):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.ANALOG_GAIN, settings["analog_gain"])
        camera.set_camera_settings(sl.VIDEO_SETTINGS.DIGITAL_GAIN, settings["digital_gain"])
    
    wb = settings['wb']
    if (wb < 2800 or wb > 6500):
        camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
    else:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)
    return

def publishNT(camera, cam_w_pose, objects, classes):
    global heartbeatPub, ntHeartbeat
    transArr = []
    # xVelArr = []
    # yVelArr = []
    # zVelArr = []
    idArr = []
    labelArr = []
    boxArr = []
    confArr = []
    isVisArr = []
    isMovArr = []

    heartbeatPub.set(ntHeartbeat)
    ntHeartbeat+=1
    latencyPub.set(camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds() - objects.timestamp.get_nanoseconds())
    camPoseLatencyPub.set(camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds() - cam_w_pose.timestamp.get_nanoseconds())
    camPoseConfPub.set(cam_w_pose.pose_confidence / 100.0) 

    objList = objects.object_list
    for obj in objList:
        idArr.append(obj.id)
        confArr.append(obj.confidence/100.0)
        isVisArr.append(obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
        isMovArr.append(obj.action_state == sl.OBJECT_ACTION_STATE.MOVING)
        labelArr.append(classes[obj.raw_label]["label"])
        pos = obj.position
        transArr.append(geom.Translation3d(-pos[2], -pos[0], pos[1]))
        # vel = obj.velocity
        # xVelArr.append(vel[0])
        # yVelArr.append(vel[1])
        # zVelArr.append(vel[2])
        dims = obj.dimensions
        boxArr.append(geom.Translation3d(dims[0], dims[2], dims[1]))

    labelPub.set(labelArr)
    idPub.set(idArr)
    confPub.set(confArr)
    isVisPub.set(isVisArr)
    isMovingPub.set(isMovArr)
    transPub.set(transArr)
    # xVelPub.set(xVelArr)
    # yVelPub.set(yVelArr)
    # zVelPub.set(zVelArr)
    boxPub.set(boxArr)
    ntInst.flush()
    wpiPose = slPoseToWPILib(cam_w_pose)
    camPosePub.set(wpiPose)
    # camOriginPub.set()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default="settings.yaml", help='settings.yaml path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')#svorecord.svo2
    opt = parser.parse_args()

    # with torch.no_grad():
    #     main()
    main()
