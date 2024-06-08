import cv2
import os
import numpy as np

# Name of the video file
vdo_name = "camera_0028"

# Capture the video from the specified file
vidcap = cv2.VideoCapture(vdo_name+'.mp4')
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
vidcap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# Read the first frame from the video
success, image = vidcap.read()
count = 0
success = True

# Loop to read frames from the video
while success:
    success, image = vidcap.read()
    if success:
        img_path = vdo_name+'_det'
        isExist = os.path.exists(img_path)
        # Create directory if it does not exist
        if not isExist:
            os.mkdir(img_path)

        # Save frame as JPEG file
        cv2.imwrite(img_path + "/frame%d.jpg" % count, image)
        # Exit if Escape is hit
        if cv2.waitKey(10) == 27:
            break
        count += 1

# Dictionary to store detection data
det_map = {}
det = open(vdo_name + ".txt")

# Read detection data from the file
for line in det:
    s = line.split(" ")
    frameId = int(float(s[0])) - 2
    x = int(float(s[3]))
    y = int(float(s[4]))
    w = int(float(s[5]))
    h = int(float(s[6]))
    bbox = [x, y, w, h, int(float(s[2]))]
    if frameId not in det_map:
        det_map[frameId] = [bbox]
    else:
        det_map[frameId].append(bbox)

# Loop through each frame ID with detection data
for fID in det_map.keys():
    fpath = "./" + vdo_name + "_det/"
    fpath1 = "./" + vdo_name + "_speed/"
    fname = "frame" + str(fID)

    isExist = os.path.exists(fpath1)
    # Create directory if it does not exist
    if not isExist:
        os.mkdir(fpath1)

    # Read the image from the saved frames
    img = cv2.imread(fpath + fname + ".jpg")

    if fID > 0:
        # Draw bounding boxes and class ID on the image
        for box in det_map[fID]:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cid = box[4]

            image = cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.putText(image, str(cid), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # Save the modified image
        cv2.imwrite(fpath1 + fname + '.jpg', img)

# Path to the folder containing modified images
image_folder = "./" + vdo_name + "_speed/"
video_name = vdo_name + '_speed_fix_dz.avi'

images = []
# List of image filenames to be included in the video
for i in range(0, 3599):  # number of frames in your video
    images.append('frame' + str(i) + '.jpg')

# Read the first frame to get the video dimensions
try:
    frame = cv2.imread(os.path.join(image_folder, images[1]))
except:
    frame = cv2.imread(os.path.join(fpath, images[1]))

height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 11.0, (width, height))

# Write each image frame to the video file
for image in images:
    if os.path.isfile(os.path.join(image_folder, image)):
        video.write(cv2.imread(os.path.join(image_folder, image)))
    else:
        video.write(cv2.imread(os.path.join(fpath, image)))

# Release the video writer and close all windows
cv2.destroyAllWindows()
video.release()
