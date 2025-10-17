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
from hl2ss.viewer import hl2ss_lnm
from hl2ss.viewer import hl2ss_mp
from hl2ss.viewer import hl2ss_3dcv
from ultralytics import YOLO
import logging
import common_functions as cf
import onnxruntime as ort
import math
import time

print("Starting script")
# Settings --------------------------------------------------------------------

# HoloLens settings
host = 'Offline'
port_left  = hl2ss.StreamPort.RM_VLC_LEFTFRONT
port_right = hl2ss.StreamPort.RM_VLC_RIGHTFRONT
calibration_path = 'calibration'
buffer_size = 10

#AI model parameters
logging.getLogger('ultralytics').setLevel(logging.ERROR)
modelpathBounding = "models/bestCropping_New_Small.pt"
modelBounding = YOLO(modelpathBounding, task='detect')

#modelpathSuper = "models/Chain.onnx"
modelpathSuper = "models/Super_90kImages_800000.onnx"
modelSuper = ort.InferenceSession(modelpathSuper)
modelNameSuper = modelSuper.get_inputs()[0].name

modelpathPose = "models/Keypose_90kImages_run16_best.pt"
modelPose = YOLO(modelpathPose)  # Replace with the path to your YOLOv8 pose model

# Initial parameters of cube
position = [0, 0, 0]
rotation = [0, 0, 0, 1]
scale = [0.05, 0.05, 0.05]
rgba = [1, 1, 1, 1]

# Functions ------------------------------------------------------------------------------

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

#------------------------------------------------------------------------------

if (__name__ == '__main__'):
    enable = True

    # Get calibration matrixes -------------------------------------------------
    calibration_lf = hl2ss_3dcv.get_calibration_rm(host, port_left, calibration_path)
    calibration_rf = hl2ss_3dcv.get_calibration_rm(host, port_right, calibration_path)
    rotation_lf = hl2ss_3dcv.rm_vlc_get_rotation(port_left)
    rotation_rf = hl2ss_3dcv.rm_vlc_get_rotation(port_right)
    K1, Rt1 = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_lf.intrinsics, calibration_lf.extrinsics, rotation_lf)
    K2, Rt2 = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_rf.intrinsics, calibration_rf.extrinsics, rotation_rf)
    
    P1 = hl2ss_3dcv.rignode_to_camera(Rt1) @ hl2ss_3dcv.camera_to_image(K1)
    P2 = hl2ss_3dcv.rignode_to_camera(Rt2) @ hl2ss_3dcv.camera_to_image(K2)



    for i in range(3):
        start_time = time.time()

        #load images from files
        image_l = np.load('savedFrames/left_vlc_raw.npy')
        image_r = np.load('savedFrames/right_vlc_raw.npy')

        image_l = hl2ss_3dcv.rm_vlc_rotate_image(image_l, rotation_lf)
        image_r = hl2ss_3dcv.rm_vlc_rotate_image(image_r, rotation_rf)
        image_l = hl2ss_3dcv.rm_vlc_to_rgb(image_l)
        image_r = hl2ss_3dcv.rm_vlc_to_rgb(image_r)

        #search for bounding box and crop the image
        start_boundingBox = time.time()
        x_l, y_l, width_l, height_l = cf.BoundingBox(image_l, modelBounding)
        imageCroppedL = cf.CropImage(image_l, x_l, y_l, width_l, height_l)

        x_r, y_r, width_r, height_r = cf.BoundingBox(image_r, modelBounding)
        imageCroppedR = cf.CropImage(image_r, x_r, y_r, width_r, height_r)
        end_boundingBox = time.time()

        #perform super resolution
        start_super = time.time()
        imageSuperL = cf.SuperResolution(imageCroppedL, modelSuper, modelNameSuper)
        imageSuperR = cf.SuperResolution(imageCroppedR, modelSuper, modelNameSuper)
        end_super = time.time()

        start_pose = time.time()
        keypointsL = cf.KeypointDetection(imageSuperL, modelPose)
        keypointsR = cf.KeypointDetection(imageSuperR, modelPose)
        end_pose = time.time()

        keypointsLFullL = cf.PointInCropToFull((x_l, y_l), (width_l, height_l), (image_l.shape[1], (image_l.shape[0])), keypointsL, normalized=False)
        keypointsLFullR = cf.PointInCropToFull((x_r, y_r), (width_r, height_r), (image_r.shape[1], (image_r.shape[0])), keypointsR, normalized=False)

        points_3d = cf.KeyPointsTo3D(P1, P2, keypointsLFullL.T, keypointsLFullR.T)
        points_3d[2, :] *= -1 #To left handed unity coordinates by flipping z ax

        circle1 = points_3d[:, 4]  
        circle8 = points_3d[:, 11]
        circle57 = points_3d[:, 60]
        circle64 = points_3d[:, 67]



        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_time_bounding = end_boundingBox - start_boundingBox
        elapsed_time_super = end_super - start_super
        elapsed_time_pose = end_pose - start_pose

        fps = 1/elapsed_time
        print("Time total: ", elapsed_time, "(fps:", fps, ")", "\n boundingbox: " , elapsed_time_bounding, " ", int(elapsed_time_bounding/elapsed_time*100),  "%\n super: ", elapsed_time_super, " ", int(elapsed_time_super/elapsed_time*100), "%\n Pose: ", elapsed_time_pose, " ", int(elapsed_time_pose/elapsed_time*100), "%")

        #for plotting
        imageCroppedL = cf.resize_image(imageCroppedL, target_size=640)
        imageSuperL = cf.resize_image(imageSuperL, target_size=640)
        imagePoseL = cf.resize_image(imageSuperL, target_size=640)
        imagePoseL = cf.ShowPoseOnImage(imagePoseL, keypointsL)
        imageCroppedAndPoseL = cf.ShowPoseOnImage(imageCroppedL, keypointsL)
        leftImages = np.hstack((image_l, imageCroppedAndPoseL, imagePoseL))

        imageCroppedR = cf.resize_image(imageCroppedR, target_size=640)
        imageSuperR = cf.resize_image(imageSuperR, target_size=640)
        imagePoseR = cf.resize_image(imageSuperR, target_size=640)
        imagePoseR = cf.ShowPoseOnImage(imagePoseR, keypointsR)
        imageCroppedAndPoseR = cf.ShowPoseOnImage(imageCroppedR, keypointsR)
        rightImages = np.hstack((image_r, imageCroppedAndPoseR, imagePoseR))
        
        differenceSize = leftImages.shape[1] - rightImages.shape[1]

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

# Create a window and set it to be resizable
cv2.namedWindow('allImages', cv2.WINDOW_NORMAL)
cv2.imshow('allImages', bothImages)
cv2.waitKey(0)
cv2.destroyAllWindows()