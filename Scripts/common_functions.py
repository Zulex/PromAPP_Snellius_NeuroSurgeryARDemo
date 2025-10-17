import multiprocessing as mp
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import logging
import csv
import time
from datetime import datetime


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


def BoundingBox(image, boundingboxModel):
    highest_confidence_bbox = None
    highest_confidence = 0

    results = boundingboxModel(image, imgsz=640)

    for r in results:  
        for i, box in enumerate(r.boxes.xywhn):
            confidence = round(float(r.boxes.conf[i]), 2)
            if confidence > highest_confidence:
                highest_confidence = confidence
                highest_confidence_bbox = box

    if highest_confidence_bbox is not None:
        x_center = highest_confidence_bbox[0]
        y_center = highest_confidence_bbox[1]
        width = highest_confidence_bbox[2]
        height = highest_confidence_bbox[3]

    return x_center, y_center, width, height

def CropImage(image, x, y, width, height):

    # Convert normalized coordinates to pixel values
    image_height, image_width, channels = image.shape
    x_center_pixel = x * image_width
    y_center_pixel = y * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    if(True):
        width_pixel = width_pixel + width_pixel / 6
        height_pixel = height_pixel + height_pixel / 6
    


    # Calculate the top-left and bottom-right coordinates of the bounding box
    left = int(x_center_pixel - width_pixel / 2)
    top = int(y_center_pixel - height_pixel / 2)
    right = int(x_center_pixel + width_pixel / 2)
    bottom = int(y_center_pixel + height_pixel / 2)

    # Crop the image using the bounding box
    cropped_image = image[top:bottom, left:right]

    return cropped_image

    
def SuperResolution(image, session, superresModelName):
    # Convert the image to a numpy array
    image_data = np.array(image).astype(np.float32)
    
    # Normalize the image (assuming normalization range [0, 1])
    image_data = image_data / 255.0
    
    # Transpose the image to match the input shape (N, C, H, W)
    image_data = np.transpose(image_data, (2, 0, 1))
    
    # Add a batch dimension
    image_data = np.expand_dims(image_data, axis=0)

    #Run superresolution model
    output_data = session.run(None, {superresModelName: image_data})[0]

    # Remove the batch dimension
    output_data = np.squeeze(output_data, axis=0)
    
    # Transpose the image to (H, W, C)
    output_data = np.transpose(output_data, (1, 2, 0))
    
    # Clip the values to [0, 1]
    output_data = np.clip(output_data, 0, 1)
    
    # Convert to uint8
    output_image = (output_data * 255).astype(np.uint8)

    # Check if the image is a proper NumPy array
    if not isinstance(output_image, np.ndarray):
        raise TypeError("The image is not a NumPy array.")
    # Ensure the image has the correct layout and type
    if len(output_image.shape) == 3 and output_image.shape[2] == 3:
        # The image is in the correct format
        pass
    else:
        raise ValueError("The image does not have the correct layout or type.")
    


    return output_image

    
def resize_image(image, target_size=640, axes = 0):
    height, width = image.shape[:2]
    
    if axes == 0:
        new_height = target_size
        new_width = int((target_size / height) * width)
    else:
        new_width = target_size
        new_height = int((target_size / width) * height)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def preprocess_image_for_superresolution(input_image):
    # Check the size of the input image
    height, width = input_image.shape[:2]
    
    # Resize if necessary
    if height < 256 or width < 256:
        input_image = cv2.resize(input_image, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Normalize if necessary (example: scale to [0, 1] if required by your model)
    input_image = input_image / 255.0

    return input_image


def KeypointDetection(image, model):
    results = model.predict(image)[0]
    if(results is not None):
        keypoints = results.keypoints.xyn[0].tolist()
    else:
        print("no pose found")
    
    return keypoints

def ShowPoseOnImage(image, keypoints, showIndex = True):
    #keypoints should be normalized
    height, width = image.shape[:2]
    color=(0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for i, keypoint in enumerate(keypoints):
        x = int(keypoint[0] * width)
        y = int(keypoint[1] * height)
        cv2.circle(image, (x, y), 3, color, -1)
        if(showIndex):
            cv2.putText(image, str(i- 3), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def pad_image(image, target_height, target_width):
    """Pad the image to the target height and width."""
    height, width = image.shape[:2]
    pad_height = target_height - height
    pad_width = target_width - width

    # Pad the image: pad_height and pad_width are added symmetrically
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Add padding
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def is_divisible_by_2(number):
    """Check if the number can be divided by 2 and remain an integer (i.e., is even)."""
    return number % 2 == 0
 


def PointInCropToFull(center_coords, crop_size, full_image_size, crop_keypoints, normalized = True):
    c_x_n, c_y_n = center_coords
    w_n, h_n = crop_size
    W, H = full_image_size

    # Convert normalized center and size to pixel values
    c_x = c_x_n * W
    c_y = c_y_n * H
    w = w_n * W
    h = h_n * H

    # Calculate the top-left corner of the crop
    x = c_x - w / 2
    y = c_y - h / 2

    full_image_keypoints = []
    for (u_c, v_c) in crop_keypoints:
        # Convert normalized coordinates to pixel coordinates in the crop
        p_x = u_c * w
        p_y = v_c * h

        # Convert pixel coordinates in the crop to pixel coordinates in the full image
        P_x = p_x + x
        P_y = p_y + y

        if(normalized):
            # Normalize the pixel coordinates in the full image
            U = P_x / W
            V = P_y / H
        else:
            U = P_x 
            V = P_y 
            
        full_image_keypoints.append((float(U), float(V)))

    return np.array(full_image_keypoints)


def KeyPointsTo3D(P1, P2, p1, p2):
    #p1 is a 2xn matrix of normalized points
    points_4d  = cv2.triangulatePoints(P1[:,:3].T, P2[:,:3].T, p1, p2)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d

def Show3DPoints():
    print("")

def process_string(input_string):
    try:
        if isinstance(input_string, bytes):
            input_string = input_string.decode('utf-8')  # Decode the bytes to a string


        # Step 1: Remove the first occurrence of 'b'
        input_string = input_string.replace('b', '')
        
        # Step 2: Remove all single quotes
        input_string = input_string.replace("'", "")
        
        # Step 3: Split the string at the semicolon
        parts = input_string.split(';')
        
        # Step 4: Convert the substrings to floats
        try:
            float_values = [float(part) for part in parts]
        except ValueError:
            raise ValueError("One of the parts cannot be converted to a float")
        
        # Step 5: Store the values in a NumPy array
        float_array = np.array(float_values)
        
        return float_array, True
    except: 
        return None, False

# Function to save data to a CSV file in batches
def save_batch_to_csv(filename, batch_data):
    """
    Writes a batch of data to the specified CSV file.

    Parameters:
    filename (str): Name of the CSV file to write to.
    batch_data (np.ndarray): A 2D numpy array where each row is [index, timestamp, data1, data2, ...].
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in batch_data:
            writer.writerow(row)

# Function to generate a timestamp
def get_timestamp():
    """Returns the current time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

