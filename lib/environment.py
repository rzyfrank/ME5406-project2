import cv2
# import gym
import numpy as np

from .create_track import create_track

class Env():
    def __init__(self, args, turning_point):
        self.args = args
        self.turning_point = turning_point
        self.track_size = (self.args.track_h, self.args.track_w)
        self.car_image = cv2.imread(args.car_path)
        self.car_image = cv2.resize(self.car_image, (10,10))

        self.car_h, self.car_w,_ = self.car_image.shape

        self.topview_size = self.args.top_view_size

        self.front_h = self.args.front_view_h
        self.front_w = self.args.front_view_w

        self.generate_track()

        self.x = self.start_x = self.start_point[0]
        self.y = self.start_y = self.start_point[1]
        self.angle = self.start_angle = self.angles[0]
        self.top_view, self.front_view = self.get_view(self.x, self.y, self.angle)
        self.speed = 0


        self.done = False
        self.gray = cv2.cvtColor(self.track_map, cv2.COLOR_BGR2GRAY)

    def generate_track(self):
        """
        Create the track by turning point
        """
        self.track_map, self.angles, self.start_point, self.ckpt, self.final_ckpt = create_track(self.track_size,
                                                                                                 self.args.track_width, self.turning_point)
        self.checkpoint()

    def checkpoint(self):
        """
        We create chekpoint for each turning
        """
        self.ckp = dict()
        for i, point in enumerate(self.ckpt):
            self.ckp[point] = (i+1) * 2

    def rotate_image(self,image, angle):
        """
        We can rotate the angle of the image to simulate the steering of a car
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def get_view(self, x, y, angle):
        """
        Once we get the position and angle of the car, we can obtain the front view and top view of the car
        """
        top_view = self.track_map[y-self.topview_size:y+self.topview_size, x-self.topview_size:x+self.topview_size]
        top_view = self.rotate_image(top_view, angle)

        a_h, a_w, _ = top_view.shape
        c_h, c_w = self.car_h, self.car_w
        x_offset = int((a_w - c_w) / 2)
        y_offset = int((a_h - c_h) / 2)
        top_view[y_offset:y_offset + c_h, x_offset:x_offset + c_w] = self.car_image

        front_view = top_view[int(a_h/2-self.front_h/2):int(a_h/2+self.front_h/2), int(a_w/2):int(a_w/2+self.front_w)]

        return top_view, front_view


    def get_position(self,x, y,angle,speed, action):
        """
        Update the angle and position by the action
        """
        angle = angle + action[0]
        angle_pi = angle/180 * np.pi
        speed = max(5, speed + action[1])   # There is a minimum speed of the car
        x += speed * np.cos(angle_pi)
        y += speed * np.sin(angle_pi)
        return int(x), int(y), angle, speed

    def reset(self):
        """
        Reset the environment for a new episode
        """
        self.generate_track()
        self.num_step = 0
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0
        self.top_view, self.front_view = self.get_view(self.x, self.y, self.angle)

        self.render_info_episode = []
        self.render_topview_episode = []
        self.render_frontview_episode = []
        self.render_info_episode.append((self.x, self.y, self.angle))
        self.render_topview_episode.append(self.top_view)
        self.render_frontview_episode.append(self.front_view)

        self.done = False

    def check_outside(self, n):
        """
        Check if the car is running off the track
        """
        if self.past_x == self.x and self.past_y == self.y:
            return False
        start_point = np.array([self.past_x, self.past_y])
        end_point = np.array([self.x, self.y])
        diff = end_point - start_point
        length = np.linalg.norm(diff)
        direction = diff / length
        points = []
        for i in range(n):
            t = i / (n - 1)
            point = start_point + t * length * direction
            points.append((int(point[1]), int(point[0])))
        points = np.array(points)
        h, w = self.gray.shape
        gray_values = self.gray.reshape((h * w,))
        indices = np.ravel_multi_index(points.T, (h, w))
        black_indices = np.where(gray_values[indices] < 5)[0]
        black_points = points[black_indices]
        if len(black_points) < 1:
            return False
        else:
            return True

    def is_finish(self):
        """
        Check if the car reaches the end of the track
        """
        if np.sqrt((self.x - self.final_ckpt[0])**2 + (self.y - self.final_ckpt[1])**2) <= self.args.track_width/2:
            return True
        else:
            return False

    def is_checkpoint(self):
        """
        Check if the car reach the checkpoint. Car can only get the reward for the checkpoint once in an episode
        """
        if self.ckp:
            for point in self.ckp.keys():
                if np.sqrt((self.x - point[1])**2 + (self.y - point[0])**2) <= (self.args.track_width)/2:
                    score = self.ckp[point]
                    del self.ckp[point]
                    return score

    def render(self):
        """
        Save information
        """
        self.render_info_episode.append((self.x, self.y, self.angle))
        self.render_topview_episode.append(self.top_view)
        self.render_frontview_episode.append(self.front_view)


    def step(self, action):
        """
        Step the car by action
        """
        self.num_step += 1
        self.past_x, self.past_y = self.x, self.y
        self.x, self.y, self.angle, self.speed = self.get_position(self.x, self.y, self.angle, self.speed, action)
        self.top_view, self.front_view = self.get_view(self.x, self.y, self.angle)

        reach = self.is_finish()
        outside = self.check_outside(self.args.split)
        score = self.is_checkpoint()

        self.render()

        if reach:
            self.done = True
            return 10
        else:
            if outside:
                self.done = True
                return -10 + np.sqrt((self.past_x - self.start_x)**2 + (self.past_y - self.start_y)**2) * 0.005
            else:
                if score:
                    self.done = False
                    return score -0.01
                else:
                    self.done = False
                    return -0.01

