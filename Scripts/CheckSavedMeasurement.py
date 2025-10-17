import os
import cv2
import numpy as np
import common_functions as cf

# Function to resize images to have the same dimensions
def resize_images_to_match(img1, img2, img3, img4):
    # Find the minimum height and width across all images
    min_height = min(img1.shape[0], img2.shape[0], img3.shape[0], img4.shape[0])
    min_width = min(img1.shape[1], img2.shape[1], img3.shape[1], img4.shape[1])

    # Resize all images to match the minimum height and width
    img1 = cv2.resize(img1, (min_width, min_height))
    img2 = cv2.resize(img2, (min_width, min_height))
    img3 = cv2.resize(img3, (min_width, min_height))
    img4 = cv2.resize(img4, (min_width, min_height))

    return img1, img2, img3, img4

# Function to display two images side by side
def display_images_side_by_side(img1, img2, img3, img4):

    img1, img2, img3, img4 = resize_images_to_match(img1, img2, img3, img4)

    # Concatenate images side by side
    combined_img1 = np.hstack((img1, img2))
    combined_img2 = np.hstack((img3, img4))

    all_img = np.vstack((combined_img1, combined_img2))

    # Display the combined image
    cv2.imshow('Left and Right Images', all_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to read images and numpy files for a given index
def process_files(indexRun, indexImage):
    # Define folder path
    foldername = f"Measurements/run_{indexRun}/"

    # Check if folder exists
    if not os.path.exists(foldername):
        print(f"Folder {foldername} does not exist!")
        return

    # Define the file paths for images and numpy arrays
    imageCropL_path = os.path.join(foldername, f"cropL_{indexImage}.png")
    imageCropR_path = os.path.join(foldername, f"cropR_{indexImage}.png")
    imageSuperL_path = os.path.join(foldername, f"superL_{indexImage}.png")
    imageSuperR_path = os.path.join(foldername, f"superR_{indexImage}.png")
    keypointL_path = os.path.join(foldername, f"keypointL_{indexImage}.npy")
    keypointR_path = os.path.join(foldername, f"keypointR_{indexImage}.npy")

    # Read the left and right images
    if os.path.exists(imageCropL_path) and os.path.exists(imageCropR_path):
        imgCropL = cv2.imread(imageCropL_path)
        imgCropR = cv2.imread(imageCropR_path)
        imgSuperL = cv2.imread(imageSuperL_path)
        imgSuperR = cv2.imread(imageSuperR_path)
    else:
        print(f"Images for index {indexImage} not found.")
        return

    # Read the numpy files for keypoints
    if os.path.exists(keypointL_path) and os.path.exists(keypointR_path):
        keypointL = np.load(keypointL_path)
        keypointR = np.load(keypointR_path)
        print(f"Successfully read numpy files: keypointL_{indexImage}.npy and keypointR_{indexImage}.npy")
    else:
        print(f"Numpy files for index {indexImage} not found.")
        return


    imgCropL = cf.resize_image(imgCropL, target_size=640)
    imgCropR = cf.resize_image(imgCropR, target_size=640)
    imgSuperL = cf.resize_image(imgSuperL, target_size=640)
    imgSuperR = cf.resize_image(imgSuperR, target_size=640)

    imgCropL = cf.ShowPoseOnImage(imgCropL, keypointL)
    imgCropR = cf.ShowPoseOnImage(imgCropR, keypointR)
    imgSuperL = cf.ShowPoseOnImage(imgSuperL, keypointL)
    imgSuperR = cf.ShowPoseOnImage(imgSuperR, keypointR)
    # Display images side by side
    display_images_side_by_side(imgCropL, imgCropL, imgSuperL, imgSuperR)

# Example usage:
indexRun = 2  # Starting index
indexImage = 20
process_files(indexRun, indexImage)
