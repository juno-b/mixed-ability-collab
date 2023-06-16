import math
import numpy as np

"""
Code from https://github.com/Yejining/I-VTFilter
"""

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_tuple(self):
        return self.x, self.y, self.z

    def get_length(self):
        return np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))

    def subtract(self, point):
        return Point3D(self.x - point.x, self.y - point.y, self.z - point.z)

    @staticmethod
    def normalize_vector(sample, point):
        vector = Point3D(sample.x - point.x, sample.y - point.y, sample.z - point.z)
        length = vector.get_length()
        return Point3D(vector.x / length, vector.y / length, vector.z / length)

    @staticmethod
    def get_angle(point1, point2):
        cosine = Point3D.dot_product(point1, point2) / (point1.get_length() * point2.get_length())
        cosine = min(1, max(-1, cosine))
        return math.acos(cosine)

    @staticmethod
    def dot_product(point1, point2):
        return (point1.x * point2.x) + (point1.y * point2.y) + (point1.z * point2.z)


def get_angular_distance(user_origin, start, end):
    start_vector = Point3D.normalize_vector(user_origin, start)
    end_vector = Point3D.normalize_vector(user_origin, end)

    if math.isnan(start_vector.x) or math.isnan(end_vector.x):
        return 0

    angle_radian = Point3D.get_angle(start_vector, end_vector)
    angle_degree = math.degrees(angle_radian)
    return angle_degree

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_length(self):
        return np.sqrt(np.power(self.x, 2) + np.power(self.y, 2))

    @staticmethod
    def get_distance(point1, point2):
        subtracted = Point2D.subtract(point1, point2)
        return math.sqrt(math.pow(subtracted.x), 2, math.pow(subtracted.y), 2)

    @staticmethod
    def subtract(point1, point2):
        return Point2D(point1.x - point2.x, point1.y - point2.y)

    @staticmethod
    def get_angle(point1, point2):
        cosine = Point2D.dot_product(point1, point2) / (point1.get_length() * point2.get_length())
        cosine = min(1, max(-1, cosine))
        return math.acos(cosine)

    @staticmethod
    def dot_product(point1, point2):
        return (point1.x * point2.x) + (point1.y * point2.y)