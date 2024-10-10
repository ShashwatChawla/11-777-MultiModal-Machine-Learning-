import cv2
import os

# Path to the folder containing images
input_folder = '/ocean/projects/cis220039p/schawla1/tartanvo/results/KITTI_10/image_left/'
flow_folder = '/ocean/projects/cis220039p/schawla1/tartanvo/results/kitti_tartanvo_1914_flow'
image_folder = flow_folder

video_name = 'kitti-10_flow.avi'

# Get the list of images
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images.sort()  # Sort the images if necessary

# Read the first image to get the width and height
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))  # 30 FPS

# Write images to the video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release the VideoWriter object
video.release()
cv2.destroyAllWindows()
