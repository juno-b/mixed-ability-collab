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
from mpl_toolkits.mplot3d import Axes3D
from Point23D import Point3D
from Data import FREQUENCY, BASIC_THRESHOLD, IVTData, GazeData, set_user_origin, AnalyzedData

#find the eye tracker
eyetrackers = tr.find_all_eyetrackers()
eyetracker = eyetrackers[0]

#get screen resolution
monitor = get_monitors()[0]
width = monitor.width
height = monitor.height

#glbal variable declaration- lists of coordinates for the left and right eyes as well as interpolated data
gaze_data_list, left_x, left_y, right_x, right_y, inter_x, inter_y, centroids_x, centroids_y = [], [], [], [], [], [], [], [], []

# I-VT filter parameters
velocity_threshold = 30  # ADJUST AS NEEDED
time_75 = 75000          # 75 ms to microseconds
time_60 = 60000          # 60 ms to microseconds

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
def write_to_csv(data_to_write, combined_fixation_data):
    #main data csv
    headers = list(data_to_write[0].keys())
    headers.extend(['selected_eye', 'inter_gaze_point_on_display_area', 'inter_gaze_origin_validity', 'inter_gaze_origin_in_trackbox_coordinate_system', 'inter_gaze_origin_in_user_coordinate_system'])
    with open('output.csv', 'w', newline = '') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_to_write)
    
    #combined fixations csv
    #headers2 = list(combined_fixation_data[0].keys())
    #headers2.extend(['selected_eye', 'inter_gaze_point_on_display_area', 'inter_gaze_origin_validity', 'inter_gaze_origin_in_trackbox_coordinate_system', 'inter_gaze_origin_in_user_coordinate_system'])#, 'time_fixation_ended'])
    headers2 = list(['id', 'time', 'velocity', 'type', 'point'])
    with open('combined_fixations.csv', 'w', newline = '') as file2:
        writer = csv.DictWriter(file2, fieldnames=headers2)
        writer.writeheader()
        for gaze_data in combined_fixation_data:
            data_dict = {
                'id': gaze_data.id,
                'time': gaze_data.time,
                'velocity': gaze_data.velocity,
                'type': gaze_data.movement_type,
                'point': gaze_data.point.to_tuple()
            }
            writer.writerow(data_dict)

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
        gaze_point = None
        #Determine if there is data available from the dominant eye
        #use this to set gaze_point as the point on the display area from the appropriate eye
        if dominantEye == 'left':
            if gaze_data['left_gaze_point_validity'] == 1:
                gaze_point = [gaze_data['left_gaze_point_on_display_area'], gaze_data['left_gaze_origin_in_trackbox_coordinate_system'], gaze_data['left_gaze_origin_in_user_coordinate_system']]
                interpolatedGazeData[i]['selected_eye'] = 'left'
            elif gaze_data['right_gaze_point_validity'] == 1:
                gaze_point = [gaze_data['right_gaze_point_on_display_area'], gaze_data['right_gaze_origin_in_trackbox_coordinate_system'], gaze_data['right_gaze_origin_in_user_coordinate_system']]
                interpolatedGazeData[i]['selected_eye'] = 'right'
        else:
            if gaze_data['right_gaze_point_validity'] == 1:
                gaze_point = [gaze_data['right_gaze_point_on_display_area'], gaze_data['right_gaze_origin_in_trackbox_coordinate_system']], gaze_data['right_gaze_origin_in_user_coordinate_system']
                interpolatedGazeData[i]['selected_eye'] = 'right'
            elif gaze_data['left_gaze_point_validity'] == 1:
                gaze_point = [gaze_data['left_gaze_point_on_display_area'], gaze_data['left_gaze_origin_in_trackbox_coordinate_system']], gaze_data['left_gaze_origin_in_user_coordinate_system']
                interpolatedGazeData[i]['selected_eye'] = 'left'
        if gaze_point is not None:
            if prev_valid_index is not None:
                prev_valid_gaze_data = interpolatedGazeData[prev_valid_index]

                # Calculate the time gap between the current and previous valid points
                delta_t = abs(gaze_data['device_time_stamp'] - prev_valid_gaze_data['device_time_stamp'])

                #we also need to interpolate the origin in the trackbox and user coordinate systems
                if delta_t <= time_75:
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

#This function is from Yejining's GitHub and calls functions from Data and Point23D to classify fixations/saccades
def classify_gaze_data(gaze_datas, frequency, threshold):
    ivt_data = IVTData(gaze_datas, frequency, threshold)

    while True:
        if ivt_data.is_iterating():
            ivt_data.calculate_angular_distance()
        ivt_data.set_window()
        if ivt_data.is_over(): break
        if ivt_data.is_window_more_than_a_second():
            ivt_data.set_velocity_and_type()

    #ivt_data.sort_velocities()
    return ivt_data.gaze_datas, ivt_data.velocities

def apply_ivt_filter(dominantEye):
    interpolatedGazeData = interpolateData(dominantEye)
    filtered_data_list = []
    #prev_valid_timestamp = interpolatedGazeData[0]['device_time_stamp']
    #prev_position = None
    if (dominantEye == 'left') & interpolatedGazeData[0]['left_gaze_origin_validity'] == 1:
        prev_position = interpolatedGazeData[0]['left_gaze_origin_in_trackbox_coordinate_system']
    elif (dominantEye == 'right') & interpolatedGazeData[0]['right_gaze_origin_validity'] == 1:
        prev_position = interpolatedGazeData[0]['right_gaze_origin_in_trackbox_coordinate_system']
    
    parsed_data = []
    for i, gaze_data in enumerate(interpolatedGazeData):
        user_pos, screen_pos = None, None

        #determine x, y, and z in the trackbox system as a tuple position
        if dominantEye == 'left':
            if gaze_data['left_gaze_origin_validity'] == 1:
                position = gaze_data['left_gaze_origin_in_trackbox_coordinate_system']
                user_pos = gaze_data['left_gaze_origin_in_user_coordinate_system']
                screen_pos = gaze_data['left_gaze_point_on_display_area']
            elif gaze_data['right_gaze_origin_validity'] == 1:
                position = gaze_data['right_gaze_origin_in_trackbox_coordinate_system']
                user_pos = gaze_data['right_gaze_origin_in_user_coordinate_system']
                screen_pos = gaze_data['right_gaze_point_on_display_area']
            elif gaze_data['inter_gaze_origin_validity'] == 1:
                position = gaze_data['inter_gaze_origin_in_trackbox_coordinate_system']
                user_pos = gaze_data['inter_gaze_origin_in_user_coordinate_system']
                screen_pos = gaze_data['inter_gaze_point_on_display_area']
        else:
            if gaze_data['right_gaze_origin_validity'] == 1:
                position = gaze_data['right_gaze_origin_in_trackbox_coordinate_system']
                user_pos = gaze_data['right_gaze_origin_in_user_coordinate_system']
                screen_pos = gaze_data['right_gaze_point_on_display_area']
            elif gaze_data['left_gaze_origin_validity'] == 1:
                position = gaze_data['left_gaze_origin_in_trackbox_coordinate_system']
                user_pos = gaze_data['left_gaze_origin_in_user_coordinate_system']
                screen_pos = gaze_data['left_gaze_point_on_display_area']
            elif gaze_data['inter_gaze_origin_validity'] == 1:
                position = gaze_data['inter_gaze_origin_in_trackbox_coordinate_system']
                user_pos = gaze_data['inter_gaze_origin_in_user_coordinate_system']
                screen_pos = gaze_data['inter_gaze_point_on_display_area']

        if((screen_pos is not None) & (user_pos is not None)):
            user_p3d = Point3D(user_pos[0], user_pos[1], user_pos[2])
            set_user_origin(user_p3d)
            parsed_data.append(GazeData(i, gaze_data['device_time_stamp'], None, Point3D(screen_pos[0]*width, screen_pos[1]*height, user_pos[2])))

    #calculate moving window velocity for each point
    classified_data, velocities = classify_gaze_data(parsed_data, FREQUENCY, BASIC_THRESHOLD)
    analyzed_data = AnalyzedData(classified_data, velocities)
    analyzed_data.init_datas()
    iterator = 0
    for vel in velocities:
        iterator += 1
    print("i:", iterator)
    print("# centroids:", len(analyzed_data.centroids))
    for cen in analyzed_data.centroids:
        centroids_x.append(cen.x)
        centroids_y.append(cen.y)
        print("centroid:", [cen.x, cen.y])
  
    return filtered_data_list, interpolatedGazeData, classified_data, analyzed_data

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
def update(frame):
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
filtered_data, interpolatedData, classified_data, analyzed_data = apply_ivt_filter(dominantEye)
#draw_pixels(filtered_data, 'Filtered')
draw_unfiltered('Unfiltered')
plot_trackbox_data(interpolatedData, 'Trackbox Coordinate System', 'left_gaze_origin_in_trackbox_coordinate_system', 'inter_gaze_origin_in_trackbox_coordinate_system')
plot_trackbox_data(interpolatedData, 'User Coordinate System', 'left_gaze_origin_in_user_coordinate_system', 'inter_gaze_origin_in_user_coordinate_system')
graph(centroids_x, centroids_y, 'Centroids')
write_to_csv(interpolatedData, classified_data)

#TASKS
#get additional data from the tobii sdk struct

#live vector IVT filter

#week 3 task- investigate gaps in coverage e.g. edges or corners of the screen

#accessibility DOM for screen readers
#bounding boxes for elements, search the hierarchy looking for the x y location and report that
#example applications: Chrome, Edge, Visual Studio