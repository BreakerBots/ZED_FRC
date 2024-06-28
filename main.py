import pyzed.sl as sl;

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # Quit if an error occurred
        exit()

        # Define the Object Detection module parameters
    detection_parameters = sl.ObjectDetectionParameters()
    # detection_parameters.image_sync = True
    detection_parameters.enable_tracking = True
    # detection_parameters.enable_mask_output = True

    # Object tracking requires camera tracking to be enabled
    if detection_parameters.enable_tracking:
        zed.enable_positional_tracking()
        
    print("Object Detection: Loading Module...")
    err = zed.enable_object_detection(detection_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}, exit program".format(err))
        zed.close()
        exit()
    # Set runtime parameter confidence to 40
    detection_parameters_runtime = sl.ObjectDetectionRuntimeParameters()
    detection_parameters_runtime.detection_confidence_threshold = 40

    objects = sl.Objects()

    # Grab new frames and detect objects
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, detection_parameters_runtime)

        if objects.is_new:
            # Count the number of objects detected
            print("{} Object(s) detected".format(len(objects.object_list)))
            objects.la
            if len(objects.object_list):
                # Display the 3D location of an object
                first_object = objects.object_list[0]
                position = first_object.position
                print(" 3D position : [{0},{1},{2}]".format(position[0],position[1],position[2]))

                # Display its 3D bounding box coordinates
                bounding_box = first_object.bounding_box
                print(" Bounding box 3D :")
                for it in bounding_box:            
                    print(" " + str(it), end='')

    zed.disable_object_detection()
    zed.close()



if __name__ == "__main__":
    main()