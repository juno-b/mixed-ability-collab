"""

This is an implementation of the Tobii I-VT Fixation Filter, which is a fixation classification algorithm
This is *not* a saccade classification algorithm/ I-VT classification filter
Based on Olsen (2012) The Tobii I-VT Fixation Filter accessed from http://www.vinis.co.kr/ivt_filter.pdf
Tested with the Tobii Pro Fusion eye tracker

@author Juno Bartsch in collaboration with Andrew Begel and Joon Jang at the Carnegie Mellon University VariAbility Lab
Portions of the code modified from https://github.com/Yejining/I-VTFilter
Last modified June 2023

Modifications:
Instead of using left, right, average, or strict average eye selection, this code uses the participant's dominant eye
If this data is not available, it will look for the other eye and then for interpolated data (which is created only
in the absence of data from either eye).

"""

import tobii_research as tr
import time
from screeninfo import get_monitors
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fa
import numpy as np
from scipy.spatial import ConvexHull
import math
from Point23D import Point3D
from Point23D import get_angular_distance
import sys
from getDom import DomObjectRetriever, setCoords
import matplotlib.image as mpimg

# Set the angle filter amount for the I-VT filter in the filter_centroids fn
# Check if command-line argument exists
if len(sys.argv) > 1:
    # Use the command-line argument
    angle_cap = float(sys.argv[1])
else:
    # default from I-VT specification is .5
    angle_cap = 0.65

#find the eye tracker
eyetrackers = tr.find_all_eyetrackers()
eyetracker = eyetrackers[0]
FREQUENCY = 250 #Hz

#get screen resolution
monitor = get_monitors()[0]
width = monitor.width
height = monitor.height

# I-VT filter parameters, ADJUST AS NEEDED
velocity_threshold = 30                     # maximum angle to be considered a fixation, default 30 degrees
maximum_interpolation_time_micro = 75000    # maximum allowed time for interpolation in microseconds
maximum_time_between_fixations = 75000      # maximumm allowed time between fixations in microseconds
maximum_angle_between_fixations = angle_cap # maximum angle between fixations in degrees, default from I-VT specification is .5
minimum_fixation_duration = 60000           # minimum fixation duration in microseconds
window_size_seconds = 0.01    # maximum time on either side of the spanning window for velocity calculations, default 10 ms --> 0.01 seconds

#global variable declaration- lists of coordinates for the left and right eyes as well as interpolated data
gaze_data_list, left_x, left_y, right_x, right_y, inter_x, inter_y, centroids_x, centroids_y = [], [], [], [], [], [], [], [], []
unfiltered_centroids_x, unfiltered_centroids_y = [], []
global_gaze_index, num_values_to_interpolate, prev_valid_point, prev_valid_time, prev_valid_idx, flag_interpolation = 0, 0, None, -1, -1, False

#Switch based on the dominant eye of the participant
dominant_eye = 'left'
#dominantEye = 'right'
     
#set selected traits from the gaze data list based on the dominant eye
if dominant_eye == 'left':
    selected_poda = 'left_gaze_point_on_display_area'
    selected_gova = 'left_gaze_origin_validity'
    selected_gotcs = 'left_gaze_origin_in_trackbox_coordinate_system'
    selected_goucs = 'left_gaze_origin_in_user_coordinate_system'
else:
    selected_poda = 'right_gaze_point_on_display_area'
    selected_gova = 'right_gaze_origin_validity'
    selected_gotcs = 'right_gaze_origin_in_trackbox_coordinate_system'
    selected_goucs = 'right_gaze_origin_in_user_coordinate_system'
inter_poda = 'inter_gaze_point_on_display_area'
inter_gova = 'inter_gaze_origin_validity'
inter_gotcs = 'inter_gaze_origin_in_trackbox_coordinate_system'
inter_goucs = 'inter_gaze_origin_in_user_coordinate_system'

#This function completes basic linear interpolation for 2 and 3-variable arrays
def linear_interpolation(indices, criteria):
    global gaze_data_list, prev_valid_time
    #calculate the start and end points, time difference, and slope
    delta_t = gaze_data_list[indices[-1]]['device_time_stamp'] - prev_valid_time
    startx = gaze_data_list[indices[0]][criteria][0]
    starty = gaze_data_list[indices[0]][criteria][1]
    endx = gaze_data_list[indices[-1]][criteria][0]
    endy = gaze_data_list[indices[-1]][criteria][1]
    return_list = []
    x_slope = (endx  - startx) / delta_t
    y_slope = (endy - starty) / delta_t
    dt = 0
    # Fill in the missing data points as a line drawn between the valid points
    if(len(gaze_data_list[0][criteria])) == 2:
        for i in range(indices[0] + 1, indices[-1]):
            dt = gaze_data_list[i]['device_time_stamp']-prev_valid_time
            x_inter = x_slope * dt + startx
            y_inter = y_slope * dt + starty
            return_list.append([x_inter, y_inter])
    else:
        startz = gaze_data_list[indices[0]][criteria][2]
        endz = gaze_data_list[indices[-1]][criteria][2]
        z_slope = (endz - startz) / delta_t
        # Fill in the missing data points as a line drawn between the valid points
        for i in range(indices[0] + 1, indices[-1]):
            dt = gaze_data_list[i]['device_time_stamp']-prev_valid_time
            x_inter = x_slope * dt + startx
            y_inter = y_slope * dt + starty
            z_inter = z_slope * dt + startz
            return_list.append([x_inter, y_inter, z_inter])
    return return_list

# Function to perform linear interpolation on the gaze data
def interpolate_gaze_data(start, current):
    global gaze_data_list
    indices = range(start, current)
    #call the linear interpolation function for the desired variables (e.g. gaze on display area, location in trackbox/user coordinate systems)
    interpolated_data1, interpolated_data2, interpolated_data3 = linear_interpolation(indices, selected_poda), linear_interpolation(indices, 
        selected_gotcs), linear_interpolation(indices, selected_goucs)
    i = 0
    for x in indices:
        #this if statement excludes the valid points used for the interpolation
        if(math.isnan(gaze_data_list[x][selected_poda][0])):
            gaze_data_list[x]['selected_eye'] = 'inter'
            gaze_data_list[x][inter_gova] = 1
            gaze_data_list[x][inter_poda] = interpolated_data1[i]
            gaze_data_list[x][inter_gotcs] = interpolated_data2[i]
            gaze_data_list[x][inter_goucs] = interpolated_data3[i]
            i+=1
            #print('live interpolation at index ' + str(x), gaze_data_list[x][inter_poda])

#This function checks if interpolation needs to occur, if so it calls the interpolation function
def check_interpolation(gaze_data):
    global prev_valid_point, prev_valid_time, prev_valid_idx, flag_interpolation, gaze_data_list
    # Interpolation works as follows: maintain an index of the last valid data point.
    # If nans are encountered and another valid data point occurs within the time threshold, interpolate the data
    timestamp = gaze_data['device_time_stamp']
    time_check = (prev_valid_time != -1) and (timestamp - prev_valid_time) < maximum_interpolation_time_micro
    # Process interpolation if criteria are met- there must be valid data to interpolate and the time criteria must be met
    # The flag for interpolation is set based on the time criteria and availability of previous data
    if(not time_check):
        flag_interpolation = False
    if(not math.isnan(gaze_data[selected_poda][0])):
        if time_check:
            if prev_valid_point is not None:
                flag_interpolation = True
                #indices are used to specify where the helper function should interpolate data
                idx = gaze_data['index']
                if (idx-prev_valid_idx>1):
                    interpolate_gaze_data(prev_valid_idx, idx+1)
                    flag_interpolation = False
        #set the previous valid point whenever a valid point is discovered
        prev_valid_point = gaze_data
        prev_valid_time = timestamp
        prev_valid_idx = gaze_data['index']

#This callback function adds gaze data from the eye tracker to the global gaze_data_list and interpolates the data live
def gaze_data_callback(gaze_data):
    #append the gaze data to the list
    gaze_data_list.append(append_pixel_data(gaze_data))
    #check if interpolation needs to happen for this data point
    check_interpolation(gaze_data)

#This function opens te eye tracker for the specified duration and then closes the connection
def run_eyetracker(duration):
    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    time.sleep(duration)
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)

#This function modifies the gaze data to be in pixels rather than fractional values and adds the helper selected eye variable
def append_pixel_data(gaze_data):
    global global_gaze_index
    #the selected eye variable is set to the dominant eye if data is available, inter if interpolated, and none otherwise
    gaze_data['selected_eye'] = 'none'
    gaze_data['index'] = global_gaze_index
    global_gaze_index += 1
    #these variables are adjusted to pixel values based on the height and width of the screen
    lx, ly = gaze_data['left_gaze_point_on_display_area']
    rx, ry = gaze_data['right_gaze_point_on_display_area']
    if (not(math.isnan(rx))):
       rx, ry = rx * width, ry * height
       gaze_data['right_gaze_point_on_display_area'] = [rx, ry]
       if dominant_eye == 'right':
           gaze_data['selected_eye'] = dominant_eye
    if (not math.isnan(lx)):
        lx, ly = lx * width, ly * height
        gaze_data['left_gaze_point_on_display_area'] = [lx, ly]
        if dominant_eye == 'left':
            gaze_data['selected_eye'] = dominant_eye

#This function writes data to a csv file. Additional data column header values should be added to headers/headers2.extend as necessary
def write_to_csv(data_to_write):#, centroid_data):
    #main data csv
    headers = list(data_to_write[1].keys())
    headers.extend(['inter_gaze_point_on_display_area', 'inter_gaze_origin_validity', 'inter_gaze_origin_in_trackbox_coordinate_system', 'inter_gaze_origin_in_user_coordinate_system'])#), 'angular_distance', 'velocity'])
    with open('output.csv', 'w', newline = '') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_to_write)
    
    #centroid csv
    '''headers2 = ['id', 'start', 'end', 'x_avg', 'y_avg', 'x_list', 'y_list', 'origin']
    with open('centroids.csv', 'w', newline='') as file2:
        writer = csv.DictWriter(file2, fieldnames=headers2)
        writer.writeheader()
        for centroid in centroid_data:
            x, y = centroid.coords()
            x_list = [item * width for item in centroid.x]
            y_list = [item * height for item in centroid.y]
            writer.writerow({
                'id': centroid.id,
                'start': centroid.start,
                'end': centroid.end,
                'x_avg': x,
                'y_avg': y,
                'x_list': x_list,
                'y_list': y_list,
                'origin': centroid.origin
            })'''

#adds the index of the point within the window furthest before (window_1) and after (window_2) each gaze point
def find_points_in_window(gaze_points):
    # Loop over the gaze points
    for i, gaze_point in enumerate(gaze_points):
        current_time = gaze_point['system_time_stamp']  / 1000000
        
        # Find the point before the current point
        for j in range(i-1, -1, -1):
            time_diff = current_time - (gaze_points[j]['system_time_stamp'] / 1000000)
            if time_diff <= window_size_seconds:
                gaze_point['window_1'] = gaze_points[j]['index']
            else:
                break

        # Find the point after the current point
        for k in range(i+1, len(gaze_points)):
            time_diff = (gaze_points[k]['system_time_stamp'] / 1000000) - current_time
            if time_diff <= window_size_seconds:
                gaze_point['window_2'] = gaze_points[k]['index']
            else:
                break   
    return gaze_points

#gets the positions in the user coordinate system and on the screen for any data point, including interpolated data
def get_user_screen_pos(gaze_data):
    user_pos, screen_pos = None, None
    #determine x, y, and z in the coordinate system and the user's gaze on the display as tuples
    if dominant_eye == 'left':
        if gaze_data['left_gaze_origin_validity'] == 1:
            user_pos = gaze_data['left_gaze_origin_in_user_coordinate_system']
            screen_pos = gaze_data['left_gaze_point_on_display_area']
        elif gaze_data['inter_gaze_origin_validity'] == 1:
            user_pos = gaze_data['inter_gaze_origin_in_user_coordinate_system']
            screen_pos = gaze_data['inter_gaze_point_on_display_area']
    else:
        if gaze_data['right_gaze_origin_validity'] == 1:
            user_pos = gaze_data['right_gaze_origin_in_user_coordinate_system']
            screen_pos = gaze_data['right_gaze_point_on_display_area']
        elif gaze_data['inter_gaze_origin_validity'] == 1:
            user_pos = gaze_data['inter_gaze_origin_in_user_coordinate_system']
            screen_pos = gaze_data['inter_gaze_point_on_display_area']
    return user_pos, screen_pos

#this function finds the gaze angle for the points within the window
def gaze_angle_velocity(interpolated_gaze_data):
    for i, gaze_data in enumerate(interpolated_gaze_data):
        prev_point_user, prev_point_screen, next_point_user, next_point_screen, window_1, window_2 = None, None, None, None, None, None
        user_pos, screen_pos = get_user_screen_pos(gaze_data)
        try:
            window_1 = gaze_data['window_1']
            window_2 = gaze_data['window_2']
        except KeyError:
            continue
        if((not math.isnan(window_1)) & (not math.isnan(window_2)) & (user_pos is not None)):
            prev_point_user, prev_point_screen = get_user_screen_pos(interpolated_gaze_data[window_1])
            next_point_user, next_point_screen = get_user_screen_pos(interpolated_gaze_data[window_2])
            if((prev_point_screen is not None) and (next_point_screen is not None)):
                user_origin = Point3D(user_pos[0], user_pos[1], user_pos[2])
                prev_point = Point3D(prev_point_screen[0]*width, prev_point_screen[1]*height, user_pos[2])
                next_point = Point3D(next_point_screen[0]*width, next_point_screen[1]*height, user_pos[2])
                ang_dist = get_angular_distance(user_origin, prev_point, next_point)
                #print('Angular distance', ang_dist)
                gaze_data['angular_distance'] = ang_dist

                prev_gaze = interpolated_gaze_data[window_1]
                next_gaze = interpolated_gaze_data[window_2]
                time_diff = (next_gaze['system_time_stamp'] / 1000000) - (prev_gaze['system_time_stamp'] / 1000000)
                if time_diff != 0:
                    velocity = gaze_data['angular_distance'] / time_diff
                    gaze_data['velocity'] = velocity
    return interpolated_gaze_data

#this function uses the angle and velocity data to find centroids and calls the function to merge adjacent fixations (filter centroids)
def find_centroids(angle_velocity_data):
    unfiltered_centroids = []
    for gaze_data in angle_velocity_data:
        try:
            if gaze_data['velocity'] <= velocity_threshold:
                gaze_data_x = gaze_data['left_gaze_point_on_display_area'][0] * width if dominant_eye == 'left' else gaze_data['inter_gaze_point_on_display_area'][0] * width
                gaze_data_y = gaze_data['left_gaze_point_on_display_area'][1] * height if dominant_eye == 'left' else gaze_data['inter_gaze_point_on_display_area'][1] * height
                if((not math.isnan(gaze_data_x)) & (not math.isnan(gaze_data_y))):
                    unfiltered_centroids.append(gaze_data)
                    #print("centroid:", [gaze_data_x, gaze_data_y])
                    unfiltered_centroids_x.append(gaze_data_x)
                    unfiltered_centroids_y.append(gaze_data_y)
        except KeyError:
            continue
    return filter_centroids(unfiltered_centroids)

#This class holds centroid data and contains functions to access time/coordinate data
class CentroidData:
    def __init__(self, id, start, end, x, y, z):
        self.id = id
        self.start = start
        self.end = end
        self.x = x
        self.y = y
        self.origin = z
    #returns the difference in start and end time of the centroid
    def time(self):
        return self.end - self.start
    #returns the sum of the x coordinates of the centroid
    def sum_x(self):
        sum_x = 0
        for item in self.x:
            sum_x += item
        return sum_x
    #returns the sum of the y coordinates of the centroid
    def sum_y(self):
        sum_y = 0
        for item in self.y:
            sum_y += item
        return sum_y
    #returns the average [x,y] coordinates of the centroid in pixels
    def coords(self):
        return [self.sum_x()/len(self.x)*width, self.sum_y()/len(self.y)*height]

#This function gets the point on the gaze display for any given gaze data point
def get_point_dom_eye(point):
    centroid_x, centroid_y, z = None, None, None
    if point['selected_eye'] == 'left':
        centroid_x = point['left_gaze_point_on_display_area'][0]
        centroid_y = point['left_gaze_point_on_display_area'][1]
        z = point['left_gaze_origin_in_user_coordinate_system']
    elif point['selected_eye'] == 'inter':
        centroid_x = point['inter_gaze_point_on_display_area'][0]
        centroid_y = point['inter_gaze_point_on_display_area'][1] 
        z = point['inter_gaze_origin_in_user_coordinate_system']
    elif point['selected_eye'] == 'right':
        centroid_x = point['right_gaze_point_on_display_area'][0] 
        centroid_y = point['right_gaze_point_on_display_area'][1] 
        z = point['right_gaze_origin_in_user_coordinate_system']
    return (centroid_x, centroid_y, z)

#This function converts a gaze data point to a CentroidData object
def gaze_tuple_to_centroid_data(gaze_tuple):
    x, y, z = get_point_dom_eye(gaze_tuple)
    return CentroidData([gaze_tuple['index']], gaze_tuple['device_time_stamp'], gaze_tuple['device_time_stamp'], [x], [y], z)    

#this function merges adjacent fixations using the maximum time and angle between fixations
def filter_centroids(unfiltered_centroids):
    #convert unfiltered data to CentroidData objects
    intermediary_centroids = []
    for centroid in unfiltered_centroids:
        intermediary_centroids.append(gaze_tuple_to_centroid_data(centroid))

    filtered_centroids = []
    i = 1
    centroid_to_add = intermediary_centroids[0]
    while i < len(intermediary_centroids):
        # check if the current point is within the maximum time and angle between fixations
        centroid = intermediary_centroids[i]
        prev_centroid_time = centroid_to_add.end
        centroid_time = centroid.start
        if abs(centroid_time - prev_centroid_time) < maximum_time_between_fixations:
            # calculate angle between the last sample in the first fixation and the first sample in the second fixation
            origin = Point3D(centroid_to_add.origin[0], centroid_to_add.origin[1], centroid_to_add.origin[2])
            p1x, p1y = centroid_to_add.coords()
            point1 = Point3D(p1x, p1y, centroid_to_add.origin[2])
            p2x, p2y = centroid.coords()
            point2 = Point3D(p2x, p2y, centroid.origin[2])
            angle = get_angular_distance(origin, point1, point2)
            if angle < angle_cap : 
                # merge to centroid to add
                centroid_to_add.id += centroid.id
                centroid_to_add.end = centroid.end
                centroid_to_add.x += centroid.x
                centroid_to_add.y += centroid.y
            else:
                filtered_centroids.append(centroid_to_add)
                centroid_to_add = centroid
        else:
            filtered_centroids.append(centroid_to_add)
            centroid_to_add = centroid
        i += 1
    for value in filtered_centroids:
        coords_x, coords_y = value.coords()
        centroids_x.append(coords_x)
        centroids_y.append(coords_y)
    return list(filtered_centroids)

#calls the interpolateData, find_points_in_window, and gaze_angle functions.
#uses this data in calculate_velocity and filters the points to centroids, which are then merged. 
def apply_ivt_filter():
    interpolatedGazeData = []#interpolateData(dominantEye)
    pointsData = find_points_in_window(interpolatedGazeData)
    angleVelocityData = gaze_angle_velocity(pointsData)
    centroidData = find_centroids(angleVelocityData)
    return interpolatedGazeData, centroidData

#This function draws the unfiltered and interpolated data 
def draw_unfiltered(title, image_path):
    # Load the image
    img = mpimg.imread(image_path)
    # Plotting convex hulls with the image as the background
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, width, 0, height])
    
    # Plot the new scatter plot with the updated data
    plt.scatter(left_x, left_y, color='blue', label='Left Eye')
    plt.scatter(right_x, right_y, color='red', label='Right Eye')
    plt.scatter(inter_x, inter_y, color = 'green', label="Interpolated")

    # Set the x and y limits
    plt.xlim(0, width)
    plt.ylim(0, height)

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(title)

    plt.show()

#This function draws the data as an animation
def draw_pixels(data, title):
    fig = plt.figure()
    fa(fig, update, frames=len(data), interval=0)
    plt.title(title)
    plt.show()

#This is the update function for the animated plots
def update(filtered_data, frame):
    # Clear the previous plot
    plt.cla()

    # Get the x and y coordinates for the current frame from the filtered data
    frame_data = filtered_data[-frame:]
    frame_left_x = [data['left_gaze_point_on_display_area'][0] * width for data in frame_data]
    frame_left_y = [data['left_gaze_point_on_display_area'][1] * height for data in frame_data]
    frame_right_x = [data['right_gaze_point_on_display_area'][0] * width for data in frame_data]
    frame_right_y = [data['right_gaze_point_on_display_area'][1] * height for data in frame_data]

    # Plot the new scatter plot with the updated data
    plt.scatter(frame_left_x, frame_left_y, color='blue', label='Left Eye')
    plt.scatter(frame_right_x, frame_right_y, color='red', label='Right Eye')

    # Set the x and y limits
    plt.xlim(0, width)
    plt.ylim(0, height)

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Animated')

    plt.show()

#This is a basic graphing function using the x and y data and a title
def graph(x, y, title):
    plt.scatter(x, y, color='blue', label=title)
    # Set the x and y limits
    plt.xlim(0, width)
    plt.ylim(0, height)
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(title)
    #plot the function
    plt.show()

#This is a basic graphing function using two sets of x/y points and titles
def graph2(x1, y1, x2, y2, title1, title2, image_path):
    # Load the image
    img = mpimg.imread(image_path)
    # Plotting convex hulls with the image as the background
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, width, 0, height])
    plt.scatter(x1, y1, color='blue', label=title1)
    plt.scatter(x2, y2, color='red', label=title2)
    # Set the x and y limits
    plt.xlim(0, width)
    plt.ylim(0, height)
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(title1 + " and " + title2)
    # Plot the scatter points
    plt.show()

#Plots the 3D trackbox coordinate data 
def plot_trackbox_data(interpolated_data, title, origin, origin2):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mx, my, mz, intx, inty, intz = [], [], [], [], [], []
    for gaze_data in interpolated_data:
        tempx2, tempy2, tempz2 = gaze_data[origin]
        if not np.isnan(tempx2) and not np.isnan(tempy2) and not np.isnan(tempz2):
            mx.append(tempx2)
            my.append(tempy2)
            mz.append(tempz2)
        try:
            tempx, tempy, tempz = gaze_data[origin2]
            if not np.isnan(tempx) and not np.isnan(tempy) and not np.isnan(tempz):
                intx.append(tempx)
                inty.append(tempy)
                intz.append(tempz)
        except KeyError:
            continue
    
    # Plot the coordinates
    ax.scatter(intx, inty, intz, color='red', label = 'Interpolated')
    ax.scatter(mx, my, mz, color = 'gray', label = 'Dominant Eye')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Combine both sets of points
    x = np.concatenate([intx, mx])
    y = np.concatenate([inty, my])
    z = np.concatenate([intz, mz])

    # Set the same scaling for all axes
    max_range = np.max([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    # Show the plot
    ax.legend()
    plt.show()

#flips the directionality of a set of coordinates
def flip_y(cen_y):
    newList = []
    for y in cen_y:
        newList.append(height - y)
    return newList

def convex_hull_plot(centroid_data, title, image_path):
    convex_list = []
    plot_number = 1  # Counter for plot numbers

    # Load the image
    img = mpimg.imread(image_path)

    for centroid in centroid_data:
        x_adj = [value * width for value in centroid.x]
        y = [value * height for value in centroid.y]
        y_adj = flip_y(y)
        points = list(zip(x_adj, y_adj))
        if len(points) > 3:
            convex_list.append(ConvexHull(points))

    # Plotting convex hulls with the image as the background
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, width, 0, height])

    for hull in convex_list:
        ax.plot(hull.points[:, 0], hull.points[:, 1], 'o')
        for simplex in hull.simplices:
            ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')

        # Add plot number as text annotation
        centroid = hull.points.mean(axis=0)
        ax.text(centroid[0], centroid[1], str(plot_number), ha='center', va='center')
        plot_number += 1

    plt.title(title)
    # Set the x and y limits
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()

    return convex_list

def is_within_boundary(x, y, boundary):
    x_min, y_min, x_max, y_max = boundary
    return x_min <= x <= x_max and y_min <= y <= y_max

def check_in_bounds(x, y):
    areas_of_interest = [
        [0, 0, 1200, 50], #taskbar
        [0, 50, 1200, 140], #footer bar
        [265, 180, 715, 1080], #left column
        [730, 233, 1635, 477], #bottom right box
        [730, 493, 1635, 1080], #top right box
    ]
    area_names = [
        "Taskbar",
        "Footer Bar",
        "Left Column",
        "Bottom Right Box",
        "Top Right Box"
    ]
    for aoi, name in zip(areas_of_interest, area_names):
        if is_within_boundary(x, y, aoi):
            #print(x, y)
            print(name)

interpolated_data, centroid_data = [], []
def run_live():
    #gaze_deque_interpolation.append({"key1": "value1"})
    #gaze_deque_interpolation.next(2)
    #print(gaze_deque_interpolation)
    run_eyetracker(3)
    #append_pixel_data()
    global interpolated_data, centroid_data
    write_to_csv(gaze_data_list)
    #interpolatedData, centroidData = apply_ivt_filter(dominantEye)
    #left_y, right_y, inter_y = flip_y(left_y), flip_y(right_y), flip_y(inter_y)
    #draw_unfiltered('Unfiltered', 'images/test.png')

'''
newY = flip_y(centroids_y)
newuY = flip_y(unfiltered_centroids_y)
graph2(unfiltered_centroids_x, newuY, centroids_x, newY, 'Unfiltered Centroids', 'Filtered Centroids', 'images/test.png')
convex_list = convex_hull_plot(centroidData, 'Convex Hull', 'images/test.png')
write_to_csv(interpolatedData, centroidData)

retriever = DomObjectRetriever()
for i, x in enumerate(centroids_x):
    if x is not None:
        setCoords(x, centroids_y[i])
        root, dom_objects, topmost_dom_object = retriever.GetTopmostDomObject(x, centroids_y[i])
        print(topmost_dom_object)
        check_in_bounds(x, abs(1200-centroids_y[i]))'''

run_live()
#TASKS
#get additional data from the tobii sdk struct

#live vector IVT filter

#week 3 task- investigate gaps in coverage e.g. edges or corners of the screen

#accessibility DOM for screen readers
#bounding boxes for elements, search the hierarchy looking for the x y location and report that
#example applications: Chrome, Edge, Visual Studio