import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_track(image_size, track_width, turning_point):
    track_map = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for i in range(len(turning_point)):
        cv2.circle(track_map, turning_point[i], int(track_width/2), (0,255,0), -1)
        if i != len(turning_point)- 1:
            cv2.line(track_map, turning_point[i],turning_point[i+1], (0,255,0), track_width)
    start_point = turning_point[0]


    end_point = turning_point[-1]
    angles = []
    for i in range(len(turning_point)-1):
        point1 = turning_point[i]
        point2 = turning_point[i+1]

        v = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        uv = v / np.linalg.norm(v)
        angle = np.arctan2(uv[1], uv[0])
        angles.append(angle)
    final_ckpt = end_point

    if angles[0] == 0:
        cv2.rectangle(track_map, (int(start_point[0] - track_width / 2), int(start_point[1] - track_width / 2)),
                      (int(start_point[0] - 1), int(start_point[1] + track_width / 2)), (0, 0, 0), -1)
    else:
        cv2.rectangle(track_map, (int(start_point[0] - track_width / 2), int(start_point[1] - track_width / 2)),
                      (int(start_point[0] + track_width / 2), int(start_point[1] - 1)), (0, 0, 0), -1)

    return track_map, angles, start_point, turning_point[1:-1], final_ckpt

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def control_position(x,y,angle_now,angle_change,v):
    angle_now += angle_change
    angle = angle_now / 180 * np.pi
    x += v*np.cos(angle)
    y += v*np.sin(angle)
    return int(x),int(y), angle_now

if __name__ == '__main__':
    turning_point = [
        (400, 500),
        (700, 500),
        (700, 800),
        (1000, 500),
        (1300, 800),
        (1600, 800)
    ]
    # turning_point = [
    #     (400, 500),
    #     (700, 500),
    #     (700, 800),
    #     (1000, 800),
    #     (1200, 600),
    #     (1600, 600)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (400, 800),
    #     (600, 600),
    #     (900, 600),
    #     (1100, 800),
    #     (1400, 800)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (700, 500),
    #     (700, 800),
    #     (1000, 800),
    #     (1300, 500),
    #     (1600, 800)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (400, 800),
    #     (700, 500),
    #     (1000, 500),
    #     (1000, 800),
    #     (1300, 800)
    # ]

    image_size = (1300, 2100)
    track_map, angles, start_point, ckpt, final_ckpt = create_track(image_size, 100, turning_point)
    local_w,local_h = 350,350

    speed, angle = 0,0

    cv2.imwrite('track.jpg', track_map)
    cv2.imshow('track', track_map)
    cv2.waitKey(0)
    x, y = start_point
    angle_point = 0
    while speed != 100:
        angle = float(input('angle:'))
        speed = float(input('speed:'))
        car_image = cv2.imread('../car.png')
        car_image = cv2.resize(car_image, (10,10))

        x,y, angle_point = control_position(x,y,angle_point, angle,speed)
        print(x, y, angle_point)
        local_area = track_map[y-local_h:y+local_h, x-local_w:x+local_w]
        local_area = rotate_image(local_area, angle_point)

        c_h, c_w,_ = car_image.shape
        a_h, a_w,_ = local_area.shape

        x_offset = int((a_w - c_w) / 2)
        y_offset = int((a_h - c_h) / 2)

        local_area[y_offset:y_offset+c_h,x_offset:x_offset+c_w] = car_image

        vision_image = local_area[int(a_h/2-112):int(a_h/2+112), int(a_w/2):int(a_w/2+224)]
        cv2.imshow('area', local_area)
        cv2.waitKey(0)
        cv2.imshow('area', vision_image)
        cv2.waitKey(0)

