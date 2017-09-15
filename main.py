import cv2
import sys
# print(sys.version)
import datetime
import imutils
import time
import numpy as np
from pandas.io import wb
import os
import math
#from optical_flow_functions import *
import time
#from variables import *
import builtins
#from test._test_multiprocessing import sqr
from opencv_funcs import *
from deep_learning_networks import *


'''Projects used in the development

http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

'''

if __name__ == '__main__':
    
    
    '''Videos folder'''
    videos_folder = "./videos_test1"
    
    ''' Lists containin the number of cars identified in each video'''
    nb_cars_list = []
    
    '''Define tf sessions'''
    # #Define Faster RCNN
    sess1, net1 = defineFasterRCNN()
    #Define CNN
    sess2, input_tensor, end_points = define_cnn_session()
    
    '''loop over all videos and get their paths'''
    video_paths = []
    for root, dirs, files in os.walk(videos_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(root, file))
      
    '''USER DEFINED INPUTS ***********************************************'''
    #Execute rcnn after a number of frames
    rcnn_int = 20
    
    name = 'test' #name to initial frames used as background
    initial1 = True
    videoPath = video_paths[0]
    frame_paths = video_to_frames(videoPath, name)    
      
    # Optical flow parameters
    background = None
    min_area = 500 #Size of the rectangle to be captured by OpenCV
    delay = 8 #Number of frames that the algorithm waits before it starts 
    overlap_threshold = 0.5 #Threshold used to decide if there is overlap between two rectangles
    resize_width = 299 
    frame_height = 299
    cars_counter = 0
    
    #print(frame_paths[0])
    frame = cv2.imread(frame_paths[0],)
    size = frame.shape
    height = frame.shape[0]
    upper_limit = frame_height *0.1
    lower_limit = frame_height *0.85
    left_limit = resize_width * 0.02
    right_limit = resize_width * 0.98
    
    video_ids = []
    frame_ids = []
    coordinates_list = []
    object_counter = 0
    object_ids = []
    video_id = 0
    rcnn_boxes = []
    dest_folder = 'pngs' #folder where png files used by rcnn function will be stored
    
    #print(video_paths)
    for videoPath in video_paths:
        video_id +=1
        cars_counter = 0
        trackers = [] #list containing tacker objects. Each tracker also contains a bbox 
        bboxes = [] # each element of this array is a bbox = (10, 23, 86, 10) 
        existing_objects = []
        new_objects = []
        #print('VideoPath: ', videoPath)
        name = '_'.join(videoPath.split('\\')[-2:])
        frame_paths = video_to_frames(videoPath, name)    
        frame_id = 0
        sizes = []
        crop_count = 0
        element = 'forward'
        '''Loop over the frames of the video with step 2 forward and backwards'''
        initial = True
           
        for i in range(0,len(frame_paths),1):
            '''Get size of the vehicles '''
            frame, gray = get_and_preProcess_frame(frame_paths, i)
            sizes = get_vehicle_size(background, sizes, gray, min_area, frame, initial)
        initial = False
        
        #print('Mean_size', np.mean(sizes))
        print('video:', videoPath)
        for i in range(0,len(frame_paths),2):#./frames/frame0.png
            #print('iteration:', i)
            object_counter = 0
            frame_id += 1
            ''' Read frame image and apply grayscale and blur'''    
            frame, gray = get_and_preProcess_frame(frame_paths, i)
            
            '''Execute RCNN after a number of iterations'''
            if i % rcnn_int == 0:
                filename_png = imgs2png(frame_paths[i], dest_folder)
                rcnn_boxes = runFasterRCNN(filename_png, sess1, net1)
                rcnn_boxes_adjusted = adjust_bbox_values(rcnn_boxes)
                 
                 
                '''Code to check if vehicle box already exists and add it otherwise'''
                rccn_new_objects, existing_objects = find_new_objects(existing_objects, rcnn_boxes_adjusted, frame, overlap_threshold)
            
                      
            '''Track frames and get existing objects coordinates and trackers of the crops which are really vehicles (they go throw CNN checking)'''
            new_objects, existing_objects, boxes_in_frame, trackers, background, ok, cars_counter, object_ids = track_frames(trackers, background, gray, existing_objects, frame_paths, delay, upper_limit, lower_limit,
                                                                                                        right_limit, left_limit, min_area, i, frame, overlap_threshold, cars_counter, object_ids,
                                                                                                        sizes, sess2, input_tensor, end_points)
            
            ''' Update tracker boxes with new positions and crop some tracker boxes'''
            if i % rcnn_int == 0:   
                #print(existing_objects)
                bboxes, existing_objects, trackers = update_tracker(rccn_new_objects, new_objects, existing_objects, boxes_in_frame, trackers, crop_count, i, gray, frame, element, name)
            else:
                bboxes, existing_objects, trackers, object_ids = update_tracker([], new_objects, existing_objects, boxes_in_frame, trackers, i, gray, frame, name, frame_id, object_ids)
           
            ''' Delete boxes close to the edge and crop some existing objects'''
            existing_objects, trackers, bboxes, object_ids = delete_boxes_and_crop_vehicles(existing_objects, crop_count, i, frame, upper_limit, lower_limit, right_limit, left_limit, bboxes, trackers, name, object_ids)
              
            '''Draw bounding box and show video'''
            #draw_bounding_box(frame, bboxes, ok)
            
            crop_count +=1
    
        nb_cars_list.append(cars_counter)
    
    ''' Obtain a List containing video path and the number of cars for each video'''
    result = list(zip(video_paths, nb_cars_list)) 
    