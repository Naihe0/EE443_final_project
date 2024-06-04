import cv2
import os

def create_video_from_frames(input_folder, output_video_path, frame_rate=30):
    # Get list of all image files in the input folder
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Ensure the frames are in order

    if not images:
        print("No images found in the input folder.")
        return

    print(f"Found {len(images)} images in the input folder.")

    # Read the first image to get the dimensions
    first_image_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi files
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for idx, image in enumerate(images):
        image_path = os.path.join(input_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
        print(f"\rProcessed frame {idx + 1}/{len(images)}: {image}", end="")

    # Release the video writer
    video.release()
    print(f"\nVideo saved as {output_video_path}")

# Usage example
input_folder = "C:\\Users\\yangningrui\\Documents\\EE443_2024_Challenge\\runs\\tracking\\inference_strongsort\\vis\\camera_0008"  # Relative path to the frames folder
output_video_path = "detect.mp4"  # Output video file name
frame_rate = 30  # Adjust frame rate as needed

create_video_from_frames(input_folder, output_video_path, frame_rate)
