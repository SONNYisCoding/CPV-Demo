import carla
import random
import time
import numpy as np
import cv2

actor_list = []

IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:,:,:3]
    cv2.imshow("", i3)
    cv2.waitKey(1)  
    return i3/255.0

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
    vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=0.0))

    actor_list.append(vehicle)

    # Camera
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=1, z=2))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_img(data))

    # Time until end
    time.sleep(50)


finally:
    for actor in actor_list:
        actor.destroy()
    print('Ok')