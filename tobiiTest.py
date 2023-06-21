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
import math
from Point23D import Point3D
from Point23D import get_angular_distance

#find the eye tracker
eyetrackers = tr.find_all_eyetrackers()
eyetracker = eyetrackers[0]

#get screen resolution
monitor = get_monitors()[0]
width = monitor.width
height = monitor.height

#glbal variable declaration- lists of coordinates for the left and right eyes as well as interpolated data
gaze_data_list, left_x, left_y, right_x, right_y, inter_x, inter_y, centroids_x, centroids_y = [], [], [], [], [], [], [], [], []
unfiltered_centroids_x, unfiltered_centroids_y = [], []

# I-VT filter parameters, ADJUST AS NEEDED
velocity_threshold = 30                     # maximum angle to be considered a fixation, default 30 degrees
maximum_interpolation_time_micro = 75000    # maximum allowed time for interpolation in microseconds
maximum_time_between_fixations = 75000      # maximumm allowed time between fixations in microseconds
maximum_angle_between_fixations = 0.5       # maximum angle between fixations in degrees
minimum_fixation_duration = 60000           # minimum fixation duration in microseconds
window_size_seconds = 0.01    # maximum time on either side of the spanning window for velocity calculations, default 10 ms --> 0.01 seconds

#Switch based on the dominant eye of the participant
dominantEye = 'left'
#dominantEye = 'right'

#This callback function adds gaze data from the eye tracker to the global gaze_data_list
def gaze_data_callback(gaze_data):
    gaze_data_list.append(gaze_data)

#This function x-y coordinates of the gaze in pixels and adds them to the global lists, since the API returns them as fractional values
def append_pixel_data():
    for row in gaze_data_list:
        left_x.append(row['left_gaze_point_on_display_area'][0]*width)
        left_y.append(row['left_gaze_point_on_display_area'][1]*height)
        right_x.append(row['right_gaze_point_on_display_area'][0]*width)
        right_y.append(row['right_gaze_point_on_display_area'][1]*height)

#This function writes data to a csv file. Additional data column header values should be added to headers/headers2.extend as necessary
def write_to_csv(data_to_write, centroid_data):
    #main data csv
    headers = list(data_to_write[1].keys())
    headers.extend(['selected_eye', 'inter_gaze_point_on_display_area', 'inter_gaze_origin_validity', 'inter_gaze_origin_in_trackbox_coordinate_system', 'inter_gaze_origin_in_user_coordinate_system', 'angular_distance', 'velocity'])
    with open('output.csv', 'w', newline = '') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_to_write)
    
    #centroid csv
    headers2 = list(centroid_data[0].keys())
    with open('centroids.csv', 'w', newline = '') as file2:
        writer = csv.DictWriter(file2, fieldnames=headers2)
        writer.writeheader()
        writer.writerows(centroid_data)

#This function opens te eye tracker for the specified duration and then closes the connection
def run_eyetracker(duration):
    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    time.sleep(duration)
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)

#This function interpolates data for gaze coordinates on the screen and the eye positions in the trackbox XYZ coordinate system
def interpolateData(dominantEye):
    interpolatedGazeData = gaze_data_list.copy()
    prev_valid_index = None
    for i, gaze_data in enumerate(gaze_data_list):
        interpolatedGazeData[i]['inter_gaze_origin_validity'] = 0
        interpolatedGazeData[i]['index'] = i
        gaze_point = None
        #Determine if there is data available from the dominant eye
        #use this to set gaze_point as the point on the display area from the appropriate eye
        if (dominantEye == 'left') & (gaze_data['left_gaze_point_validity'] == 1) & (not math.isnan(gaze_data['left_gaze_point_on_display_area'][0])):
                gaze_point = [gaze_data['left_gaze_point_on_display_area'], gaze_data['left_gaze_origin_in_trackbox_coordinate_system'], gaze_data['left_gaze_origin_in_user_coordinate_system']]
                interpolatedGazeData[i]['selected_eye'] = 'left'
        elif (dominantEye == 'right') & (gaze_data['right_gaze_point_validity'] == 1) & (not math.isnan(gaze_data['right_gaze_point_on_display_area'][0])):
                gaze_point = [gaze_data['right_gaze_point_on_display_area'], gaze_data['right_gaze_origin_in_trackbox_coordinate_system']], gaze_data['right_gaze_origin_in_user_coordinate_system']
                interpolatedGazeData[i]['selected_eye'] = 'right'
        if gaze_point is not None:
            if prev_valid_index is not None:
                prev_valid_gaze_data = interpolatedGazeData[prev_valid_index]

                # Calculate the time gap between the current and previous valid points
                delta_t = abs(gaze_data['device_time_stamp'] - prev_valid_gaze_data['device_time_stamp'])

                #we also need to interpolate the origin in the trackbox and user coordinate systems
                if delta_t <= maximum_interpolation_time_micro:
                    # Calculate the slope for linear interpolation of gaze point
                    x1, y1 = prev_valid_gaze_data['left_gaze_point_on_display_area'] if dominantEye == 'left' else prev_valid_gaze_data['right_gaze_point_on_display_area']
                    x2, y2 = gaze_point[0]
                    x_slope = (x2 - x1) / delta_t
                    y_slope = (y2 - y1) / delta_t

                    # Slope for x, y, and z of the trackbox coordinate system
                    trackbox_x2, trackbox_y2, trackbox_z2 = gaze_point[1]
                    trackbox_x1, trackbox_y1, trackbox_z1 = prev_valid_gaze_data['left_gaze_origin_in_trackbox_coordinate_system']
                    if(prev_valid_gaze_data['selected_eye'] == 'right'):
                        trackbox_x1, trackbox_y1, trackbox_z1 = prev_valid_gaze_data['right_gaze_origin_in_trackbox_coordinate_system']
                    elif(prev_valid_gaze_data['selected_eye'] == 'inter'):
                        trackbox_x1, trackbox_y1, trackbox_z1 = prev_valid_gaze_data['inter_gaze_origin_in_trackbox_coordinate_system']
                    
                    tbx_slope = (trackbox_x2 - trackbox_x1) / delta_t
                    tby_slope = (trackbox_y2 - trackbox_y1) / delta_t
                    tbz_slope = (trackbox_z2 - trackbox_z1) / delta_t

                    # Slope for x, y, and z of the user coordinate system
                    user_x2, user_y2, user_z2 = gaze_point[2]
                    user_x1, user_y1, user_z1 = prev_valid_gaze_data['left_gaze_origin_in_user_coordinate_system']
                    if(prev_valid_gaze_data['selected_eye'] == 'right'):
                        user_x1, user_y1, user_z1 = prev_valid_gaze_data['right_gaze_origin_in_user_coordinate_system']
                    elif(prev_valid_gaze_data['selected_eye'] == 'inter'):
                        user_x1, user_y1, user_z1 = prev_valid_gaze_data['inter_gaze_origin_in_user_coordinate_system']
                    
                    ucx_slope = (user_x2 - user_x1) / delta_t
                    ucy_slope = (user_y2 - user_y1) / delta_t
                    ucz_slope = (user_z2 - user_z1) / delta_t
                    
                    #declaring the timestamp and distance dt
                    t = prev_valid_gaze_data['device_time_stamp']
                    dt = 0
                    # Fill in the missing data points as a line drawn between the valid points
                    for j in range(prev_valid_index + 1, i):
                        dt = gaze_data_list[j]['device_time_stamp']-t 
                        x_interpolated = x_slope * dt + x1
                        y_interpolated = y_slope * dt + y1

                        tbx_inter = tbx_slope * dt + trackbox_x1
                        tby_inter = tby_slope * dt + trackbox_y1
                        tbz_inter = tbz_slope * dt + trackbox_z1

                        ucx_inter = ucx_slope * dt + user_x1
                        ucy_inter = ucy_slope * dt + user_y1
                        ucz_inter = ucz_slope * dt + user_z1
                        
                        #add to global lists for plotting
                        inter_x.append(x_interpolated*width), inter_y.append(y_interpolated*height)

                        interpolatedGazeData[j]['inter_gaze_point_on_display_area'] = [x_interpolated, y_interpolated]
                        interpolatedGazeData[j]['selected_eye'] = 'inter'
                        interpolatedGazeData[j]['inter_gaze_origin_validity'] = 1
                        interpolatedGazeData[j]['inter_gaze_origin_in_trackbox_coordinate_system'] = [tbx_inter, tby_inter, tbz_inter]
                        interpolatedGazeData[j]['inter_gaze_origin_in_user_coordinate_system'] = [ucx_inter, ucy_inter, ucz_inter]
            prev_valid_index = i
    return interpolatedGazeData

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
    if dominantEye == 'left':
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
def gaze_angle_velocity(interpolatedGazeData):
    for i, gaze_data in enumerate(interpolatedGazeData):
        prev_point_user, prev_point_screen, next_point_user, next_point_screen, window_1, window_2 = None, None, None, None, None, None
        user_pos, screen_pos = get_user_screen_pos(gaze_data)
        try:
            window_1 = gaze_data['window_1']
            window_2 = gaze_data['window_2']
        except KeyError:
            continue
        if((not math.isnan(window_1)) & (not math.isnan(window_2)) & (user_pos is not None)):
            prev_point_user, prev_point_screen = get_user_screen_pos(interpolatedGazeData[window_1])
            next_point_user, next_point_screen = get_user_screen_pos(interpolatedGazeData[window_2])
            if((prev_point_screen is not None) and (next_point_screen is not None)):
                user_origin = Point3D(user_pos[0], user_pos[1], user_pos[2])
                prev_point = Point3D(prev_point_screen[0]*width, prev_point_screen[1]*height, user_pos[2])
                next_point = Point3D(next_point_screen[0]*width, next_point_screen[1]*height, user_pos[2])
                ang_dist = get_angular_distance(user_origin, prev_point, next_point)
                #print('Angular distance', ang_dist)
                gaze_data['angular_distance'] = ang_dist

                prev_gaze = interpolatedGazeData[window_1]
                next_gaze = interpolatedGazeData[window_2]
                time_diff = (next_gaze['system_time_stamp'] / 1000000) - (prev_gaze['system_time_stamp'] / 1000000)
                if time_diff != 0:
                    velocity = gaze_data['angular_distance'] / time_diff
                    gaze_data['velocity'] = velocity
    return interpolatedGazeData

#this function uses the angle and velocity data to find centroids and calls the function to merge adjacent fixations (filter centroids)
def find_centroids(angleVelocityData):
    unfiltered_centroids = []
    for gaze_data in angleVelocityData:
        try:
            if gaze_data['velocity'] <= velocity_threshold:
                gaze_data_x = gaze_data['left_gaze_point_on_display_area'][0] * width if dominantEye == 'left' else gaze_data['inter_gaze_point_on_display_area'][0] * width
                gaze_data_y = gaze_data['left_gaze_point_on_display_area'][1] * height if dominantEye == 'left' else gaze_data['inter_gaze_point_on_display_area'][1] * height
                if((not math.isnan(gaze_data_x)) & (not math.isnan(gaze_data_y))):
                    unfiltered_centroids.append(gaze_data)
                    print("centroid:", [gaze_data_x, gaze_data_y])
                    unfiltered_centroids_x.append(gaze_data_x)
                    unfiltered_centroids_y.append(gaze_data_y)
        except KeyError:
            continue
    return filter_centroids(unfiltered_centroids)

#this function merges adjacent fixations using the maximum time and angle between fixations
def filter_centroids(unfiltered_centroids):
    filtered_centroids = []
    potential_centroid = []
    potential_centroid.append(unfiltered_centroids[0])
    for i, centroid in enumerate(unfiltered_centroids, start=1):
        #check if the current point is within the maximum time and angle between fixations
        prev_centroid_angle = unfiltered_centroids[i-1]['angular_distance']
        prev_centroid_time = unfiltered_centroids[i-1]['device_time_stamp']
        centroid_angle = centroid['angular_distance']
        centroid_time = centroid['device_time_stamp']
        print(prev_centroid_angle, prev_centroid_time, centroid_angle, centroid_time)
        print("angle_diff:", abs(prev_centroid_angle - centroid_angle), "time_diff:", abs(prev_centroid_time - centroid_time))
        if((abs(prev_centroid_angle - centroid_angle) > maximum_angle_between_fixations) or 
            (abs(prev_centroid_time - centroid_time) > maximum_time_between_fixations)):
            print("condition triggered")
            #if the threshold is exceeded, the points from the potential centroid must be evaluated and cleared
            if len(potential_centroid) > 1:
                #check if the time between the first and last points in the potential centroid is more than the minimum fixation duration
                if (potential_centroid[-1]['device_time_stamp'] - potential_centroid[0]['device_time_stamp']) > minimum_fixation_duration:
                    #the centroid is the average of the points
                    centroid_x, centroid_y = 0, 0
                    for point in potential_centroid:
                        centroid_x += point[0]
                        centroid_y += point[1]
                    centroid_x /= len(potential_centroid)
                    centroid_y /= len(potential_centroid)
                    filtered_centroids.append([centroid_x, centroid_y])
                    #these variables are for graphing the centroids
                    centroids_x.append(centroid_x)
                    centroids_y.append(centroid_y)
            potential_centroid = []
        else:
            #otherwise, this point should be appended to the list of points considered in the combined centroid
            potential_centroid.append(centroid)
    return filtered_centroids

#calls the interpolateData, find_points_in_window, and gaze_angle functions.
#uses this data in calculate_velocity and filters the points to centroids, which are then merged. 
def apply_ivt_filter(dominantEye):
    interpolatedGazeData = interpolateData(dominantEye)
    pointsData = find_points_in_window(interpolatedGazeData)
    angleVelocityData = gaze_angle_velocity(pointsData)
    centroidData = find_centroids(angleVelocityData)
    return interpolatedGazeData, centroidData

#This function draws the unfiltered and interpolated data 
def draw_unfiltered(title):
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

#Plots the 3D trackbox coordinate data 
def plot_trackbox_data(interpolatedData, title, origin, origin2):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mx, my, mz, intx, inty, intz = [], [], [], [], [], []
    for gaze_data in interpolatedData:
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

#call the necessary functions
run_eyetracker(5)
append_pixel_data()
interpolatedData, centroidData = apply_ivt_filter(dominantEye)
draw_unfiltered('Unfiltered')
plot_trackbox_data(interpolatedData, 'Trackbox Coordinate System', 'left_gaze_origin_in_trackbox_coordinate_system', 'inter_gaze_origin_in_trackbox_coordinate_system')
plot_trackbox_data(interpolatedData, 'User Coordinate System', 'left_gaze_origin_in_user_coordinate_system', 'inter_gaze_origin_in_user_coordinate_system')
graph(centroids_x, centroids_y, 'Centroids')
write_to_csv(interpolatedData, centroidData)

#TASKS
#get additional data from the tobii sdk struct

#live vector IVT filter

#week 3 task- investigate gaps in coverage e.g. edges or corners of the screen

#accessibility DOM for screen readers
#bounding boxes for elements, search the hierarchy looking for the x y location and report that
#example applications: Chrome, Edge, Visual Studio