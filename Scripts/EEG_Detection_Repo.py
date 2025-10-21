"""
Author: Hizirwan S. Salim  
Purpose: Inference pipeline for EEG electrode localization using stereo video from HoloLens 2.  
         The script loads pre-trained bounding box, super-resolution, and keypoint detection models.  
         It processes synchronized stereo frames, performs 3D triangulation of EEG electrode positions,  
         and logs results alongside pose estimates obtained via DINO pose regression.

Usage: This script assumes the exact hardware, calibration setup, and model formats as described  
       in the associated publication. No configuration abstraction or general-purpose interface is provided.

Support: This code is provided as-is. It has been tested only with the hardware and network configuration  
         specified in the referenced paper. No official support or maintenance is offered.

Citation: If you use this work, cite the original publication:  
Salim, H.S., et al. "Super-resolution for localizing electrode grids as small, deformable objects during 
epilepsy surgery using augmented reality headsets"  
International Journal of Computer Assisted Radiology and Surgery (2025).  
DOI: https://doi.org/10.1007/s11548-025-03401-5
"""

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
import KalmanFilter 
import unityserver
import threading
import csv


print("Starting script")
# Settings --------------------------------------------------------------------

# HoloLens settings
host = '192.168.137.94' #fill in your hololens IP address here
port_left  = hl2ss.StreamPort.RM_VLC_LEFTFRONT
port_right = hl2ss.StreamPort.RM_VLC_RIGHTFRONT
calibration_path = 'calibration'
buffer_size = 10

# Models
modelBounding = YOLO("models/bestCropping_New_Small.engine", task='detect')
modelSuper = ort.InferenceSession("models/Super_90kImages_800000.onnx")
modelNameSuper = modelSuper.get_inputs()[0].name
modelPose = YOLO("models/Keypose_90kImages_run16_best.pt")

#Saving functions
ParentFolder = "Measurements/DINO/"
folderName = "run"
filename = "Test.csv"
batch_size= 25

#Introduce a warmup just to inspect data before it starts saving
warmup = True
warmupCounter = 0
warmupAmount = 20

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

def run_server():
    global server
    server = unityserver.PeerToPeerServer("192.168.137.84", 5000)
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    finally:
        print("Stopping TCP server for DINO locations")
        server.stop()


def get_unique_filename(folder, filename):
    base, extension = os.path.splitext(filename) 
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    
    return new_filename

def get_unique_foldername(parent_folder, foldername):
    base = os.path.join(parent_folder, foldername)  
    counter = 1
    new_foldername = f"{base}_{counter}" 
    while os.path.exists((new_foldername)):
        new_foldername = f"{base}_{counter}"
        counter += 1
    
    return new_foldername

def list_active_threads():
    active_threads = threading.enumerate()
    print("Active Threads:")
    for thread in active_threads:
        print(f"- {thread.name} (ID: {thread.ident}, Alive: {thread.is_alive()})")


#------------------------------------------------------------------------------

if (__name__ == '__main__'):
    enable = True
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    calibration_lf = hl2ss_3dcv.get_calibration_rm(host, port_left, calibration_path)
    calibration_rf = hl2ss_3dcv.get_calibration_rm(host, port_right, calibration_path)
    rotation_lf = hl2ss_3dcv.rm_vlc_get_rotation(port_left)
    rotation_rf = hl2ss_3dcv.rm_vlc_get_rotation(port_right)
    K1, Rt1 = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_lf.intrinsics, calibration_lf.extrinsics, rotation_lf)
    K2, Rt2 = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_rf.intrinsics, calibration_rf.extrinsics, rotation_rf)

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
    folderName = get_unique_foldername(ParentFolder, folderName)
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
        headers += [f'EEGCable{i+1}{axis}' for i in range(4) for axis in ['x', 'y', 'z']]
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
            P1 = hl2ss_3dcv.rignode_to_camera(Rt1) @ hl2ss_3dcv.camera_to_image(K1)
            P2 = hl2ss_3dcv.rignode_to_camera(Rt2) @ hl2ss_3dcv.camera_to_image(K2)
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

        try:
            #We only access NDI Tool position when also the EEG has been found to ensure same amount of frames
            if server:  # Ensure the server has been created and is running
                current_position = server.get_current_position()  # Thread-safe access to currentPosition
                DinoPosAndRot, processed = cf.process_string(current_position)
                if(not processed):
                    continue
            else:
                print("ERROR: Could not retrieve location and rotation from the NDI tools measured by the hololens.")
        except: 
            print("ERROR: not able to communicate with TCP server for DINO pose")

        if (warmup and warmupCounter < warmupAmount):
            warmupCounter = warmupCounter + 1
            continue


        points_3d[2, :] *= -1 #To left handed unity coordinates by flipping z ax
        circle1 = points_3d[:, 4]  
        circle8 = points_3d[:, 11]
        circle57 = points_3d[:, 60]
        circle64 = points_3d[:, 67]

        EEGpositions = []
        for i in range(68):
            data_x = points_3d[:, i][0]  # Data1x to Data67x
            data_y = points_3d[:, i][1]  # Data1x to Data67x
            data_z = points_3d[:, i][2]  # Data1x to Data67x

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
        
        cropLFilename = os.path.join(folderName, f"cropL_{index}.png")
        cropRFilename = os.path.join(folderName, f"cropR_{index}.png")
        superLFilename = os.path.join(folderName, f"superL_{index}.png")
        superRFilename = os.path.join(folderName, f"superR_{index}.png")
        keypointLFilename = os.path.join(folderName, f"keypointL_{index}.npy")
        keypointRFilename = os.path.join(folderName, f"keypointR_{index}.npy")
        cv2.imwrite(cropLFilename, imageCroppedL)
        cv2.imwrite(cropRFilename, imageCroppedR)
        cv2.imwrite(superLFilename, imageSuperL)
        cv2.imwrite(superRFilename, imageSuperR)
        np.save(keypointLFilename, keypointsL)
        np.save(keypointRFilename, keypointsR)
        
        # Apply dynamic Kalman filter adjustment
        for i in range(points_3d.shape[1]):
            filtered_position = KalmanFilter.dynamic_adjustment(kalman_filters[i], previous_positions[i], points_3d[:, i])
            previous_positions[i] = filtered_position  # Update previous position

        # Apply Kalman filter to smooth the 3D points
        filtered_circle1 = previous_positions[4]
        filtered_circle8 = previous_positions[11]
        filtered_circle57 = previous_positions[60]
        filtered_circle64 = previous_positions[67]

        #for plotting
        image_l = cf.resize_image(image_l, target_size=320)
        imageCroppedL = cf.resize_image(imageCroppedL, target_size=320)
        imageSuperL = cf.resize_image(imageSuperL, target_size=320)
        imagePoseL = cf.resize_image(imagePoseL, target_size=320)
        imagePoseL = cf.ShowPoseOnImage(imagePoseL, keypointsL)
        imageCroppedAndPoseL = cf.ShowPoseOnImage(imageCroppedL, keypointsL)
        leftImages = np.hstack((image_l, imageCroppedAndPoseL, imagePoseL))

        image_r = cf.resize_image(image_r, target_size=320)
        imageCroppedR = cf.resize_image(imageCroppedR, target_size=320)
        imageSuperR = cf.resize_image(imageSuperR, target_size=320)
        imagePoseR = cf.resize_image(imagePoseR, target_size=320)
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

        cv2.imshow('allImages', bothImages)
        cv2.waitKey(1)


    # Stop streams ------------------------------------------------------------
    server.stop()
    server_thread.join()
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
  