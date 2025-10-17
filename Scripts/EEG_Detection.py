#------------------------------------------------------------------------------
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pynput import keyboard
import multiprocessing as mp
import numpy as np
import cv2
from hl2ss.viewer import hl2ss


from hl2ss.viewer import hl2ss
from hl2ss.viewer import hl2ss_lnm
from hl2ss.viewer import hl2ss_mp
from hl2ss.viewer import hl2ss_3dcv
from ultralytics import YOLO
import logging
from hl2ss.viewer import hl2ss_rus
import common_functions as cf
import onnxruntime as ort
import math
import KalmanFilter  # Assuming you saved the KalmanFilter3D class in a separate file.
import unityserver
import threading
import csv



print("Starting script")
# Settings --------------------------------------------------------------------

# HoloLens settings
host = '192.168.1.55'
port_left  = hl2ss.StreamPort.RM_VLC_LEFTFRONT
port_right = hl2ss.StreamPort.RM_VLC_RIGHTFRONT
calibration_path = 'calibration'
buffer_size = 10

#AI model parameters
logging.getLogger('ultralytics').setLevel(logging.ERROR)
modelpathBounding = "models/bestCropping.engine"
modelBounding = YOLO(modelpathBounding, task='detect')

modelpathSuper = "models/Super_90kImages_800000.onnx"
modelSuper = ort.InferenceSession(modelpathSuper)
modelNameSuper = modelSuper.get_inputs()[0].name

modelpathPose = "models/Keypose_90kImages_run16_best.pt"
modelPose = YOLO(modelpathPose)  # Replace with the path to your YOLOv8 pose model

# Initial parameters of cube
position = [0, 0, 0]
rotation = [0, 0, 0, 1]
scale = [0.005, 0.005, 0.005]
rgba = [1, 1, 1, 1]

#Saving functions
folderName = "Measurements"
filename = "Test.csv"
batch_size= 25

# Functions ------------------------------------------------------------------------------

def ShowImage(image_l, image_r, pts_left, pts_right, distance):
        cv2.circle(image_l, (int(pts_left[0]), int(pts_left[1])), 2, (0, 255, 0), 2)  
        cv2.circle(image_r, (int(pts_right[0]), int(pts_right[1])), 2, (0, 255, 0), 2) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text = str(distance)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        image_height, image_width, _ = image_l.shape
        x = (image_width - text_width) // 2
        y = (image_height + text_height) // 2
        cv2.putText(image_l, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        image = np.hstack((image_l, image_r))
        cv2.imshow('Rectified', image)
        cv2.waitKey(1)

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

server = None  # Global variable to store the server instance
def run_server():
    global server  # Declare that we are modifying the global server variable
    server_host = "192.168.1.97"
    server_port = 5000
    server = unityserver.PeerToPeerServer(server_host, server_port)

    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    finally:
        server.stop()

# Create a function to handle file name checking and modification
def get_unique_filename(folder, filename):
    base, extension = os.path.splitext(filename)  # Split name and extension
    counter = 1
    new_filename = filename
    
    # Loop to find a unique filename if one already exists
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    
    return new_filename

#------------------------------------------------------------------------------

if (__name__ == '__main__'):
    enable = True
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Get calibration matrixes -------------------------------------------------
    calibration_lf = hl2ss_3dcv.get_calibration_rm(host, port_left, calibration_path)
    calibration_rf = hl2ss_3dcv.get_calibration_rm(host, port_right, calibration_path)
    rotation_lf = hl2ss_3dcv.rm_vlc_get_rotation(port_left)
    rotation_rf = hl2ss_3dcv.rm_vlc_get_rotation(port_right)
    K1, Rt1 = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_lf.intrinsics, calibration_lf.extrinsics, rotation_lf)
    K2, Rt2 = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_rf.intrinsics, calibration_rf.extrinsics, rotation_rf)


    # Start streams -----------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(port_left, hl2ss_lnm.rx_rm_vlc(host, port_left))
    producer.configure(port_right, hl2ss_lnm.rx_rm_vlc(host, port_right))
    producer.initialize(port_left, buffer_size * hl2ss.Parameters_RM_VLC.FPS)
    producer.initialize(port_right, buffer_size * hl2ss.Parameters_RM_VLC.FPS)
    producer.start(port_left)
    producer.start(port_right)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_left = consumer.create_sink(producer, port_left, manager, ...)
    sink_right = consumer.create_sink(producer, port_right, manager, None)
    sink_left.get_attach_response()
    sink_right.get_attach_response()

    #used for messaging and creating object
    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    # Initialize Kalman filters for each 3D point
    initial_positions = np.zeros((68, 3))  # Assuming 68 points, initially at origin
    kalman_filters = [KalmanFilter.KalmanFilter3D(initial_position) for initial_position in initial_positions]
    previous_positions = initial_positions.copy()


    # Start the server in a new thread
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    #for saving the data
    os.makedirs(folderName, exist_ok=True)
    filename = get_unique_filename(folderName, filename)
    full_path = os.path.join(folderName, filename)
    # Initialize CSV file with headers
    
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Customize headers for 80 data columns
        # Simplified header creation
        headers = ['Index', 'Timestamp']
        headers += [f'DinoPos{axis}' for axis in ['x', 'y', 'z']]
        headers += [f'DinoRot{axis}' for axis in ['x', 'y', 'z', 'w']]
        # Adding Data1x to Data67z dynamically
        headers += [f'EEG{i+1}{axis}' for i in range(64) for axis in ['x', 'y', 'z']]
        writer.writerow(headers)

    index = 0
    data_batch = []

    while (enable):
        # Get frames ----------------------------------------------------------
        sink_left.acquire()
        _, data_left = sink_left.get_most_recent_frame()
        if (data_left is None):
            continue

        _, data_right = sink_right.get_nearest(data_left.timestamp)
        if (data_right is None):
            continue
        
        try:
            P1 = hl2ss_3dcv.world_to_reference(data_right.pose) @ hl2ss_3dcv.rignode_to_camera(Rt1) @ hl2ss_3dcv.camera_to_image(K1)
            P2 = hl2ss_3dcv.world_to_reference(data_left.pose) @ hl2ss_3dcv.rignode_to_camera(Rt2) @ hl2ss_3dcv.camera_to_image(K2)
        except:
            print("ERROR: Hololens cannot find its own world position -> not saving results")
            continue
        

        # Using undistorted, rotated, unrectified images --------------------------
        lf_u = cv2.remap(data_left.payload,  calibration_lf.undistort_map[:, :, 0], calibration_lf.undistort_map[:, :, 1], cv2.INTER_LINEAR)
        rf_u = cv2.remap(data_right.payload, calibration_rf.undistort_map[:, :, 0], calibration_rf.undistort_map[:, :, 1], cv2.INTER_LINEAR)
        lf_ru = hl2ss_3dcv.rm_vlc_rotate_image(lf_u, rotation_lf)
        rf_ru = hl2ss_3dcv.rm_vlc_rotate_image(rf_u, rotation_rf)
        image_l = hl2ss_3dcv.rm_vlc_to_rgb(lf_ru)
        image_r = hl2ss_3dcv.rm_vlc_to_rgb(rf_ru)
        # image = np.hstack((image_l, image_r)) 


        #image_l = hl2ss_3dcv.rm_vlc_rotate_image(data_left.payload, rotation_lf)
        #image_r = hl2ss_3dcv.rm_vlc_rotate_image(data_right.payload, rotation_rf)
        #image_l = hl2ss_3dcv.rm_vlc_to_rgb(image_l)
        #image_r = hl2ss_3dcv.rm_vlc_to_rgb(image_r)

        try:
            #search for bounding box and crop the image
            x_l, y_l, width_l, height_l = cf.BoundingBox(image_l, modelBounding)
            imageCroppedL = cf.CropImage(image_l, x_l, y_l, width_l, height_l)

            x_r, y_r, width_r, height_r = cf.BoundingBox(image_r, modelBounding)
            imageCroppedR = cf.CropImage(image_r, x_r, y_r, width_r, height_r)
        except:
            bothImages = np.hstack((image_l, image_r))
            cv2.imshow('allImages', bothImages)
            cv2.waitKey(1)
            continue


        print(index, " Bounding box detected")


        imageSuperL = imageCroppedL.copy()
        imageSuperR = imageCroppedR.copy()

        #perform super resolution
        try:
            imageSuperL = cf.SuperResolution(imageSuperL, modelSuper, modelNameSuper)
            imageSuperR = cf.SuperResolution(imageSuperR, modelSuper, modelNameSuper)
        except:
            print("ERROR: there is an error in superresolution function -> Not saving results")
            continue

        imagePoseL = imageSuperL.copy()
        imagePoseR = imageSuperR.copy()
        keypointsL = cf.KeypointDetection(imagePoseL, modelPose)
        keypointsR = cf.KeypointDetection(imagePoseR, modelPose)


        keypointsLFullL = cf.PointInCropToFull((x_l, y_l), (width_l, height_l), (image_l.shape[1], (image_l.shape[0])), keypointsL, normalized=False)
        keypointsLFullR = cf.PointInCropToFull((x_r, y_r), (width_r, height_r), (image_r.shape[1], (image_r.shape[0])), keypointsR, normalized=False)

        try:
            points_3d = cf.KeyPointsTo3D(P1, P2, keypointsLFullL.T, keypointsLFullR.T)
        except:
            print("Couldn't perform pose detection given the found keyposes")
            continue

        

        #We only access NDI Tool position when also the EEG has been found to ensure same amount of frames
        if server:  # Ensure the server has been created and is running
            current_position = server.get_current_position()  # Thread-safe access to currentPosition
            DinoPosAndRot, processed = cf.process_string(current_position)
            if(not processed):
                continue
        else:
            print("ERROR: Could not retrieve location and rotation from the NDI tools measured by the hololens.")

        
        points_3d[2, :] *= -1 #To left handed unity coordinates by flipping z ax
        circle1 = points_3d[:, 4]  
        circle8 = points_3d[:, 11]
        circle57 = points_3d[:, 60]
        circle64 = points_3d[:, 67]


        EEGpositions = []
        for i in range(64):
            data_x = points_3d[:, 4 + i][0]  # Data1x to Data67x
            data_y = points_3d[:, 4 + i][1]  # Data1x to Data67x
            data_z = points_3d[:, 4 + i][2]  # Data1x to Data67x

            EEGpositions.extend([data_x, data_y, data_z])

        index = index + 1
        timestamp = cf.get_timestamp()
        allData = [index, timestamp] + list(DinoPosAndRot) + EEGpositions
        data_batch.append(allData)

        # Check if batch is full, then save it to CSV
        if len(data_batch) >= batch_size:
            print(f"Saving batch of {batch_size} rows to CSV...")
            cf.save_batch_to_csv(full_path, np.array(data_batch))
            data_batch = []  # Clear the batch after saving
        
        # Apply dynamic Kalman filter adjustment
        for i in range(points_3d.shape[1]):
            filtered_position = KalmanFilter.dynamic_adjustment(kalman_filters[i], previous_positions[i], points_3d[:, i])
            previous_positions[i] = filtered_position  # Update previous position


        # Apply Kalman filter to smooth the 3D points
        filtered_circle1 = previous_positions[4]
        filtered_circle8 = previous_positions[11]
        filtered_circle57 = previous_positions[60]
        filtered_circle64 = previous_positions[67]

        #distanceCalc1 = np.linalg.norm(circle1 - circle8)
        #distanceCalc2 = np.linalg.norm(circle1 - circle57)
        #distanceCalc3 = np.linalg.norm(circle57 - circle64)
        #distanceCalc4 = np.linalg.norm(circle8 - circle64)
        #print("distance between corners in mm: " , distanceCalc1 * 1000, distanceCalc2 *1000, distanceCalc3 *1000, distanceCalc4 *1000) # in mm



        #for plotting
        imageCroppedL = cf.resize_image(imageCroppedL, target_size=640)
        imageSuperL = cf.resize_image(imageSuperL, target_size=640)
        imagePoseL = cf.resize_image(imagePoseL, target_size=640)
        imagePoseL = cf.ShowPoseOnImage(imagePoseL, keypointsL)
        imageCroppedAndPoseL = cf.ShowPoseOnImage(imageCroppedL, keypointsL)
        leftImages = np.hstack((image_l, imageCroppedAndPoseL, imagePoseL))

        imageCroppedR = cf.resize_image(imageCroppedR, target_size=640)
        imageSuperR = cf.resize_image(imageSuperR, target_size=640)
        imagePoseR = cf.resize_image(imagePoseR, target_size=640)
        imagePoseR = cf.ShowPoseOnImage(imagePoseR, keypointsR)
        imageCroppedAndPoseR = cf.ShowPoseOnImage(imageCroppedR, keypointsR)
        rightImages = np.hstack((image_r, imageCroppedAndPoseR, imagePoseR))

        differenceSize = abs(leftImages.shape[1] - rightImages.shape[1])

        

        if(cf.is_divisible_by_2(differenceSize)):
            if(leftImages.shape[1] > rightImages.shape[1]):
                padding = int(differenceSize / 2)
                imageCroppedAndPoseR = cf.pad_image(imageCroppedAndPoseR, imageCroppedAndPoseR.shape[0], imageCroppedAndPoseR.shape[1] + padding)
                imagePoseR = cf.pad_image(imagePoseR, imagePoseR.shape[0], imagePoseR.shape[1] + padding)
                rightImages = np.hstack((image_r, imageCroppedAndPoseR, imagePoseR))
            else:
                padding = int(differenceSize / 2)
                imageCroppedAndPoseL = cf.pad_image(imageCroppedAndPoseL, imageCroppedAndPoseL.shape[0], imageCroppedAndPoseL.shape[1] + padding)
                imagePoseL = cf.pad_image(imagePoseL, imagePoseL.shape[0], imagePoseL.shape[1] + padding)
                leftImages = np.hstack((image_l, imageCroppedAndPoseL, imagePoseL))

        else:
            if(leftImages.shape[1] > rightImages.shape[1]):
                paddingA = math.floor(differenceSize / 2)
                paddingB = math.ceil(differenceSize/2) 
                imageCroppedAndPoseR = cf.pad_image(imageCroppedAndPoseR, imageCroppedAndPoseR.shape[0], imageCroppedAndPoseR.shape[1] + paddingA)
                imagePoseR = cf.pad_image(imagePoseR, imagePoseR.shape[0], imagePoseR.shape[1] + paddingB)
                rightImages = np.hstack((image_r, imageCroppedAndPoseR, imagePoseR))
            else:
                paddingA = math.floor(differenceSize / 2)
                paddingB = math.ceil(differenceSize/2) 
                imageCroppedAndPoseL = cf.pad_image(imageCroppedAndPoseL, imageCroppedAndPoseL.shape[0], imageCroppedAndPoseL.shape[1] + paddingA)
                imagePoseL = cf.pad_image(imagePoseL, imagePoseL.shape[0], imagePoseL.shape[1] + paddingB)
                leftImages = np.hstack((image_l, imageCroppedAndPoseL, imagePoseL))

        bothImages = np.vstack((leftImages, rightImages))


        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        display_list.set_world_transform(1, filtered_circle1, rotation, scale)
        display_list.set_world_transform(8, filtered_circle8, rotation, scale)
        display_list.set_world_transform(57, filtered_circle57, rotation, scale)
        display_list.set_world_transform(64, filtered_circle64, rotation, scale)

        display_list.end_display_list()
        ipc.push(display_list)
        results = ipc.pull(display_list)

        cv2.imshow('allImages', bothImages)
        cv2.waitKey(1)


    # Stop streams ------------------------------------------------------------
    server.stop()
    sink_left.detach()
    sink_right.detach()
    producer.stop(port_left)
    producer.stop(port_right)
    command_buffer = hl2ss_rus.command_buffer()
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer)
    ipc.close()

    #stop keyboard events
    listener.join()
