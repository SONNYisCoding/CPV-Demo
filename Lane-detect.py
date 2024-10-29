import carla
import random
import time
import numpy as np
import cv2
import math

actor_list = []

IM_WIDTH = 640
IM_HEIGHT = 480

def get_steering_angle(lines, image_shape):
    # Calculate steering angle based on detected lines
    if lines is None:
        return 0
    
    # Separate lines into left and right lanes
    left_lines = []
    right_lines = []
    # Need improvement
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Filter horizontal lines (need improvement)
        if abs(slope) < 0.1:
            continue
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    # If no lines detected, maintain straight direction
    if len(left_lines) == 0 and len(right_lines) == 0:
        return 0
    
    # Calculate average line for each side (need improvement)
    left_avg = np.zeros(4)
    right_avg = np.zeros(4)
    
    if len(left_lines) > 0:
        left_avg = np.mean(left_lines, axis=0)[0]
    if len(right_lines) > 0:
        right_avg = np.mean(right_lines, axis=0)[0]
    
    # Calculate center point between lanes
    center_x = image_shape[1] // 2
    
    # If both lanes detected
    if len(left_lines) > 0 and len(right_lines) > 0:
        left_x2 = left_avg[2]
        right_x2 = right_avg[2]
        lane_center = (left_x2 + right_x2) // 2
    # If only left lane detected
    elif len(left_lines) > 0:
        left_x2 = left_avg[2]
        lane_center = left_x2 + 0  # Assume lane width
    # If only right lane detected
    elif len(right_lines) > 0:
        right_x2 = right_avg[2]
        lane_center = right_x2 - 0 # Assume lane width
    else:
        lane_center = center_x
    
    # Calculate offset from center
    offset = lane_center - center_x
    
    # Convert offset to steering angle
    # Adjust this value to make steering more or less aggressive
    steering_sensitivity = 0.01 
    steering_angle = np.clip(offset * steering_sensitivity, -1.0, 1.0)
    
    return -steering_angle

def process_img(image, vehicle):
    # Convert raw data to numpy array
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    original = i2[:,:,:3].astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define region of interest
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(50, IM_HEIGHT),
                            (IM_WIDTH//2 - 45, IM_HEIGHT//2),
                            (IM_WIDTH//2 + 45, IM_HEIGHT//2),
                            (IM_WIDTH - 50, IM_HEIGHT)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Create a copy of original image for drawing
    line_image = original.copy()
    
    # Hough Transform
    lines = cv2.HoughLinesP(masked_edges,
                           rho=2,
                           theta=np.pi/180,
                           threshold=50,
                           minLineLength=40,
                           maxLineGap=100)
    
    # Draw lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Calculate steering angle
    steering_angle = get_steering_angle(lines, original.shape)
    
    # Apply control to vehicle
    throttle = 0.3  # Constant speed
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering_angle))
    
    # Draw steering indicator
    cv2.line(line_image, 
             (IM_WIDTH//2, IM_HEIGHT),
             (int(IM_WIDTH//2 + steering_angle * 100), IM_HEIGHT-50),
             (255, 0, 0), 3)
    
    # Display the result
    cv2.imshow("Lane Following", line_image)
    cv2.waitKey(1)
    
    return original/255.0

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    client.load_world('Town04_Opt')

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    checkpoints = world.get_map().get_spawn_points()
    spawn_point = checkpoints[0]

    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)

    # Camera setup
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=1, z=2))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    # Pass vehicle reference to process_img
    sensor.listen(lambda data: process_img(data, vehicle))

    time.sleep(50)

finally:
    for actor in actor_list:
        actor.destroy()
    print('Ok')
    cv2.destroyAllWindows()