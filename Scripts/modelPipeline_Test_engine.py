#------------------------------------------------------------------------------
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
import tensorrt as trt
import common
from common import *
import torch

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
modelpathBounding = "models/bestCropping_New_Small.engine"
modelBounding = YOLO(modelpathBounding, task='detect')

#modelpathSuper = "models/Chain.onnx"
modelpathSuper = "models/Super_90kImages_800000.onnx"
modelSuper = ort.InferenceSession(modelpathSuper)
modelNameSuper = modelSuper.get_inputs()[0].name

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

with open("models/Super_90kImages_800000.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engineL = runtime.deserialize_cuda_engine(f.read())

with open("models/Super_90kImages_800000.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engineR = runtime.deserialize_cuda_engine(f.read())

contextL = engineL.create_execution_context()
contextR = engineR.create_execution_context()

dynamic_input_shapes = [
                        [1,3,20,20],  # min
                        [1,3,60,60],  # opt
                        [1,3,180,180]   # max
                        ]
max_output_shapes = (1, 3, dynamic_input_shapes[2][2]*4, dynamic_input_shapes[2][3]*4)
inputsL, outputsL, bindingsL, streamL = common.allocate_buffers(engineL, max_output_shapes, profile_idx=0)
inputsR, outputsR, bindingsR, streamR = common.allocate_buffers(engineR, max_output_shapes, profile_idx=0)

dur_time = 0


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


def preprocess_image(img):
   """
   Function to preprocess the image.
   Includes color conversion, tensor transformation, and normalization.

   Parameters:
      img (np.ndarray): Input image in BGR format.

   Returns:
      np.ndarray: Preprocessed image tensor.
   """
   with torch.no_grad():
      # Convert BGR -> RGB
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # Convert HWC -> CHW and uint8 -> float32
      img = torch.from_numpy(img).permute(2, 0, 1).float()
      # Normalize to [0, 1]
      img /= 255.0
      # Add batch dimension (C, H, W) -> (1, C, H, W)
      img = img.unsqueeze(0)
      # Return as NumPy array (C-order)   
   return np.array(img, dtype=np.float32, order="C")

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



    for i in range(1):
        start_time = time.time()

        #load images from files
        image_l = np.load('savedFrames/left_vlc_raw.npy')
        image_r = np.load('savedFrames/right_vlc_raw.npy')

        start_preprocess = time.time()
        image_l = hl2ss_3dcv.rm_vlc_rotate_image(image_l, rotation_lf)
        image_r = hl2ss_3dcv.rm_vlc_rotate_image(image_r, rotation_rf)
        image_l = hl2ss_3dcv.rm_vlc_to_rgb(image_l)
        image_r = hl2ss_3dcv.rm_vlc_to_rgb(image_r)
        end_preprocess = time.time()

        #search for bounding box and crop the image
        start_boundingBox = time.time()
        x_l, y_l, width_l, height_l = cf.BoundingBox(image_l, modelBounding)
        imageCroppedL = cf.CropImage(image_l, x_l, y_l, width_l, height_l)

        x_r, y_r, width_r, height_r = cf.BoundingBox(image_r, modelBounding)
        imageCroppedR = cf.CropImage(image_r, x_r, y_r, width_r, height_r)
        end_boundingBox = time.time()

        

        #perform super resolution old
        #start_super_old = time.time()
        #imageSuperL_old = cf.SuperResolution(imageCroppedL, modelSuper, modelNameSuper)
        #imageSuperR_old = cf.SuperResolution(imageCroppedR, modelSuper, modelNameSuper)
        #end_super_old = time.time()

        #perform super resolution new
        start_super_new = time.time()
        preprocessedL = preprocess_image(imageCroppedL)
        preprocessedR = preprocess_image(imageCroppedR)

        batch_imagesL = np.concatenate([preprocessedL], axis=0)
        batch_imagesR = np.concatenate([preprocessedR], axis=0)

        inputsL[0].host = batch_imagesL
        inputsR[0].host = batch_imagesR
        contextL.set_input_shape('input', batch_imagesL.shape) 
        contextR.set_input_shape('input', batch_imagesR.shape) 

        output_shapesL = (1, 3, batch_imagesL.shape[2]*4, batch_imagesL.shape[3]*4)
        output_shapesR = (1, 3, batch_imagesR.shape[2]*4, batch_imagesR.shape[3]*4)


        trt_outputsL = common.do_inference(contextL, engine=engineL, bindings=bindingsL, inputs=inputsL, outputs=outputsL, stream=streamL)
        torch.cuda.synchronize()
        trt_outputsR = common.do_inference(contextR, engine=engineR, bindings=bindingsR, inputs=inputsR, outputs=outputsR, stream=streamR)
        torch.cuda.synchronize()


        vol = 1
        for elem in output_shapesL:
            vol *= elem
        new_outputs = trt_outputsL[0][:vol]
        t_outputsL = new_outputs.reshape(output_shapesL)

        vol = 1
        for elem in output_shapesR:
            vol *= elem
        new_outputs = trt_outputsR[0][:vol]
        t_outputsR = new_outputs.reshape(output_shapesR)

        # Step 1: Remove the batch dimension
        imageSuperL = t_outputsL[0]  # Shape becomes (3, 128, 144)
        imageSuperR = t_outputsR[0]  # Shape becomes (3, 128, 144)

        imageSuperL = ((np.transpose(imageSuperL, (1, 2, 0)))*255).astype(np.uint8)  # Shape becomes (128, 144, 3)
        imageSuperR = ((np.transpose(imageSuperR, (1, 2, 0)))*255).astype(np.uint8)  # Shape becomes (128, 144, 3)
        
        end_super_new = time.time()
  
        start_pose = time.time()
        keypointsL = cf.KeypointDetection(imageSuperL, modelPose)
        keypointsR = cf.KeypointDetection(imageSuperR, modelPose)
        end_pose = time.time()

        start_postprocess = time.time()
        keypointsLFullL = cf.PointInCropToFullSpeed((x_l, y_l), (width_l, height_l), (image_l.shape[1], (image_l.shape[0])), keypointsL, normalized=False)
        keypointsLFullR = cf.PointInCropToFullSpeed((x_r, y_r), (width_r, height_r), (image_r.shape[1], (image_r.shape[0])), keypointsR, normalized=False)

        #points_3d = cf.KeyPointsTo3D(P1, P2, keypointsLFullL.T, keypointsLFullR.T)
        #points_3d[2, :] *= -1 #To left handed unity coordinates by flipping z ax

        #circle1 = points_3d[:, 4]  
        #circle8 = points_3d[:, 11]
        #circle57 = points_3d[:, 60]
        #circle64 = points_3d[:, 67]
        end_postprocess = time.time()



        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_time_preprocess = end_preprocess - start_preprocess
        elapsed_time_postprocess = end_postprocess - start_postprocess
        elapsed_time_bounding = end_boundingBox - start_boundingBox
        elapsed_time_super = end_super_new - start_super_new
        elapsed_time_pose = end_pose - start_pose
        percetageBouding = int(elapsed_time_bounding/elapsed_time*100)
        percentageSuper = int(elapsed_time_super/elapsed_time*100)
        percentagePose = int(elapsed_time_pose/elapsed_time*100)
        percentagePreProcess = int(elapsed_time_preprocess/elapsed_time*100)
        percentagePostProcess = int(elapsed_time_postprocess/elapsed_time*100)
        
        rest = 100 - percetageBouding - percentageSuper - percentagePose

        fps = 1/elapsed_time
        #print("Time total: ", elapsed_time, "(fps:", fps, ")", "\n boundingbox: " , elapsed_time_bounding, " ", percetageBouding,  "%\n super: ", elapsed_time_super, " ", percentageSuper, "%\n Pose: ", elapsed_time_pose, " ", percentagePose, "%\n Rest: ", rest)

        print("Time total: ", elapsed_time, "(fps:", fps, ")", "\n preprocess", percentagePreProcess, "%\n boundingbox:", percetageBouding,  "%\n super: ", percentageSuper, "%\n Pose: ", percentagePose, "%\n postprocess: ", percentagePostProcess ,"%\n")

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