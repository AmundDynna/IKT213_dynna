import cv2
import time

filepath = "assignment_1/solutions/camera_outputs.txt"

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps = cam.get(cv2.CAP_PROP_FPS)

frames = 0
start_time = time.time()
while frames < 300:
    ret, frame = cam.read()
    frames += 1
total_time = time.time() - start_time
fps = frames / total_time

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()

with open(filepath, "w") as f:
    f.write(f"FPS Tested: {fps}\n")
    f.write(f"FPS Camera: {cam_fps}\n")
    f.write(f"Height: {frame_height}\n")
    f.write(f"Width: {frame_width}\n")
