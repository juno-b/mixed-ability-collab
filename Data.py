import Point23D as p3d
from Point23D import Point3D
from Point23D import Point2D
import numpy as np
import math

"""
Code from https://github.com/Yejining/I-VTFilter
"""

user_origin = Point3D(-80.10377502441406, -59.44341278076172, 708.3294677734375)

UNKNOWN = 0
FIXATION = 1
SACCADE = 2

SAVE_IMAGE = 0
PLOT_IMAGE = 1

FREQUENCY = 250
BASIC_THRESHOLD = 30

FIXATION_COLUMNS = ['average number of fixations',
                    'minimum value of fixations',
                    'maximum value of fixations',
                    'standard deviation of number of fixations',
                    'dispersion of fixation']
VELOCITY_COLUMNS = ['number of velocities',
                    'average of velocities',
                    'median',
                    'minimum',
                    'maximum',
                    'standard deviation',
                    'dispersion']

def set_user_origin(user):
    global user_origin
    print('user origin changed from:',user_origin,'to',user)
    user_origin = user

class IVTData:
    def __init__(self, gaze_datas, frequency, threshold):
        self.gaze_datas = gaze_datas
        self.frequency = frequency
        self.threshold = threshold
        self.velocities = []
        self.amplitudes = []
        self.buffer = []
        self.window = 0
        self.index = 0

    def is_iterating(self):
        return True if self.index + 1 < len(self.gaze_datas) else False

    def is_over(self):
        return True if self.window < 1 and self.index + 1 >= len(self.gaze_datas) else False

    def is_window_more_than_a_second(self):
        return True if self.window >= 1 else False

    def append_buffer(self):
        self.buffer.append(self.index)

    def append_amplitude(self):
        point1, point2, time1, time2 = self.get_last_consecutive_points()
        angular_distance = p3d.get_angular_distance(user_origin, point1, point2)
        self.amplitudes.append(angular_distance)
        print('angular distance:', angular_distance, "point1:", [point1.x, point1.y], "point2", [point2.x, point2.y], "user_origin:", [user_origin.x, user_origin.y, user_origin.z])

        eye_distance = user_origin.z
        vertical_height = abs(point1.y-point2.y)
        horizontal_distance = abs(point1.x - point2.x)
        diagonal_distance = math.sqrt(vertical_height**2 + horizontal_distance**2)
        angular_distance_rad = math.atan(diagonal_distance / eye_distance)
        angular_distance_deg = math.degrees(angular_distance_rad)
        print("Angular Distance:", angular_distance_deg)

        try:
            velocity = angular_distance_deg / ((time2-time1) / 1000000)
            print('window', self.window, 'sum', sum(self.amplitudes), 'velocity', velocity)
            self.gaze_datas[self.buffer[0]].velocity = velocity
        except ZeroDivisionError:
            pass

    def append_velocity(self):
        velocity = sum(self.amplitudes) / self.window
        #self.gaze_datas[self.buffer[0]].velocity = velocity
        self.velocities.append(velocity)

    def increase_index(self):
        self.index += 1

    def calculate_angular_distance(self):
        self.append_buffer()
        self.append_amplitude()
        self.increase_index()

    def get_velocity(self):
        return self.velocities[len(self.velocities) - 1]

    def set_velocity_and_type(self):
        self.append_velocity()
        self.set_movement_type()

    def sort_velocities(self):
        self.velocities.sort()
        list(set(self.velocities))

    def set_movement_type(self):
        velocity = self.get_velocity()
        self.gaze_datas[self.buffer[0]].movement_type = FIXATION if velocity < self.threshold else SACCADE
        self.amplitudes.pop(0)
        self.buffer.pop(0)

    def set_window(self):
        first_id, last_id = self.get_first_last_point_id()
        self.window = (last_id - first_id) / (self.frequency)

    def get_last_consecutive_points(self):
        current_index = self.buffer[len(self.buffer) - 2]
        last_index = self.buffer[len(self.buffer) - 1]
        point1 = self.gaze_datas[current_index].point
        point2 = self.gaze_datas[last_index].point
        time1 = self.gaze_datas[current_index].time
        time2 = self.gaze_datas[last_index].time
        return point1, point2, time1, time2

    def get_first_last_point_id(self):
        first_index = self.buffer[0]
        last_index = self.buffer[len(self.buffer) - 1]
        return self.gaze_datas[first_index].id, self.gaze_datas[last_index].id

class GazeData:
    def __init__(self, id, time, velocity, point):
        self.id = id
        self.time = time
        self.velocity = velocity
        self.point = point
        self.movement_type = 0

    @staticmethod
    def subtract_time(start, end):
        return end - start

class AnalyzedData:
    def __init__(self, gaze_datas, velocities):
        self.gaze_datas = gaze_datas
        self.velocities = velocities
        self.fixations = []
        self.saccades = []
        self.centroids = []
        self.centroids_lines = []
        self.fixation_index = 0

    def init_datas(self):
        while self.fixation_index < len(self.gaze_datas) - 1:
            if self.gaze_datas[self.fixation_index].movement_type is FIXATION:
                centroid = self.centroid_of_fixation()
                self.centroids.append(centroid)
            elif self.gaze_datas[self.fixation_index].movement_type is SACCADE:
                self.set_saccades()
            else:
                break
        self.set_centroids_lines()

    def centroid_of_fixation(self):
        count = 0
        x_sum = 0
        y_sum = 0
        while self.gaze_datas[self.fixation_index].movement_type is FIXATION:
            x_sum += self.gaze_datas[self.fixation_index].point.x
            y_sum += self.gaze_datas[self.fixation_index].point.y
            self.fixations.append(self.gaze_datas[self.fixation_index].point)
            self.fixation_index += 1
            count += 1
        return Point2D(x_sum / count, y_sum / count)

    def set_saccades(self):
        while self.gaze_datas[self.fixation_index].movement_type is SACCADE:
            self.saccades.append(self.gaze_datas[self.fixation_index].point)
            self.fixation_index += 1

    def set_centroids_lines(self):
        lines = []
        for i in range(len(self.centroids) - 1):
            start = [self.centroids[i].x, self.centroids[i].y]
            end = [self.centroids[i + 1].x, self.centroids[i + 1].y]
            lines.append([start, end])
        array_line = np.array(lines)
        self.to_point(array_line)

    def to_point(self, array_line):
        if len(array_line) == 0: return

        x_list = array_line[:, :, 0].T
        y_list = array_line[:, :, 1].T
        for i in range(len(x_list)):
            self.centroids_lines.append(Point2D(x_list[i], y_list[i]))