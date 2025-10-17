#------------------------------------------------------------------------------
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard
import multiprocessing as mp
import numpy as np
import cv2
import hl2ss
from hl2ss.viewer import hl2ss_lnm
from hl2ss.viewer import hl2ss_mp
from hl2ss.viewer import hl2ss_3dcv
from ultralytics import YOLO
from PIL import Image
import logging
from hl2ss.viewer import hl2ss_rus

print("Starting script")
# Settings --------------------------------------------------------------------

# HoloLens settings
host = '192.168.1.55'
port_left  = hl2ss.StreamPort.RM_VLC_LEFTFRONT
port_right = hl2ss.StreamPort.RM_VLC_RIGHTFRONT
calibration_path = 'calibration'
buffer_size = 10

#AI model parameters
print("loading model")
logging.getLogger('ultralytics').setLevel(logging.ERROR)
modelpath = "models/bestCropping.engine"
model = YOLO(modelpath, task='detect')

# Initial parameters of cube
position = [0, 0, 0]
rotation = [0, 0, 0, 1]
scale = [0.05, 0.05, 0.05]
rgba = [1, 1, 1, 1]

# Functions ------------------------------------------------------------------------------

def MeasureDistance(image_l, image_r, P1in, P2in):
    
    p1 = np.zeros([2,1], dtype= float)
    p2 = np.zeros([2,1], dtype= float)

    try:
        #predict bounding boxes
        results_l = model(image_l, imgsz=640)
        results_r = model(image_r, imgsz=640)

    #extract centers of bounding boxes
        counterp1 = 0
        
        for box_idx, l in enumerate(results_l):
            for bbox in l.boxes:
                x_l, y_l, w, h = map(int, bbox.xywh[0].tolist())
                if (counterp1 == 0):
                    p1[0] = x_l 
                    p1[1] = y_l 
                counterp1 = counterp1 + 1

        counterp2 = 0
        for box_idx, r in enumerate(results_r):
            for bbox in r.boxes:
                x_r, y_r, w, h = map(int, bbox.xywh[0].tolist())
                if (counterp2 == 0):
                    p2[0] = x_r 
                    p2[1] = y_r 
                counterp2 = counterp2 + 1

        points_4d  = cv2.triangulatePoints(P1in[:,:3].T, P2in[:,:3].T, p1, p2)
        x1 = points_4d[:3, 0] / points_4d[3, 0]

        T = np.empty([3, 1], dtype=float)
        distanceCalc = np.linalg.norm(x1 - T)
        return distanceCalc, p1, p2, x1

    except NameError as e :
        print("Except: {e}")
        distanceCalc = 0
        x1 = np.zeros([3,1], dtype=float)
        return distanceCalc, p1, p2, x1


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
    key = 0
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube) # Create a cube, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the cube
    display_list.set_world_transform(key, position, rotation, scale) # Set the world transform of the cube
    display_list.set_color(key, rgba) # Set the color of the cube
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the cube visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    key = results[2] # Get the cube id, created by the 3rd command in the list

    print(f'Created cube with id {key}')



    while (enable):
        # Get frames ----------------------------------------------------------
        sink_left.acquire()
        _, data_left = sink_left.get_most_recent_frame()
        if (data_left is None):
            continue

        _, data_right = sink_right.get_nearest(data_left.timestamp)
        if (data_right is None):
            continue


        P1 = hl2ss_3dcv.world_to_reference(data_right.pose) @ hl2ss_3dcv.rignode_to_camera(Rt1) @ hl2ss_3dcv.camera_to_image(K1)
        P2 = hl2ss_3dcv.world_to_reference(data_left.pose) @ hl2ss_3dcv.rignode_to_camera(Rt2) @ hl2ss_3dcv.camera_to_image(K2)
        image_l = hl2ss_3dcv.rm_vlc_rotate_image(data_left.payload, rotation_lf)
        image_r = hl2ss_3dcv.rm_vlc_rotate_image(data_right.payload, rotation_rf)
        image_l = hl2ss_3dcv.rm_vlc_to_rgb(image_l)
        image_r = hl2ss_3dcv.rm_vlc_to_rgb(image_r)

        distance, pts_left, pts_right, loc1 = MeasureDistance(image_l, image_r, P1, P2)
        ShowImage(image_l, image_r, pts_left, pts_right, distance)
        loc1[2] = loc1[2] * -1
        position[0] = loc1[0]
        position[1] = loc1[1]
        position[2] = loc1[2]
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        display_list.set_world_transform(key, position, rotation, scale)
        display_list.end_display_list()
        ipc.push(display_list)
        results = ipc.pull(display_list)


    # Stop streams ------------------------------------------------------------
    sink_left.detach()
    sink_right.detach()
    producer.stop(port_left)
    producer.stop(port_right)
    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove(key) # Destroy cube
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer)
    ipc.close()

    #stop keyboard events
    listener.join()
