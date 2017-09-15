import cv2
import sys
import datetime
import imutils
import time
import numpy as np
from pandas.io import wb
import os
import math
import time
import builtins


from deep_learning_networks import *


def rect_area(x1,x2,y1,y2):
    return np.abs(x1-x2) * np.abs(y1-y2)

def find_boxes_in_frame(background, gray, upper_limit, lower_limit, right_limit, left_limit, min_area, frame, sizes, sess2, input_tensor, end_points):
    '''This function returns a list where each item corresponds to the coordinates of a box where openCV detects pixel motion'''
    # compute the absolute difference between the current frame and
    # previous + delay frame
    #threshold is 25 --> If the delta is less than 25, we discard the pixel and set it to black 
    frameDelta = cv2.absdiff(background, gray)
    thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, cnts, hierarchy= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes_in_frame = []
    # loop over the contours
    for c in cnts:
        min_area = 10
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            #if not it starts at the beginning in the next iteration
            continue    
        '''find close contour'''
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        try:
            resizer = y/255 * 10
            sqrt_mean_vehicle = int(math.sqrt(np.mean(sizes)*resizer))
        except Exception:
            resizer = 1
            sqrt_mean_vehicle = 40
        if cv2.contourArea(c) < 500:
            w = sqrt_mean_vehicle
            h = sqrt_mean_vehicle 
            x = x - int(sqrt_mean_vehicle/4)
            y = y - int(sqrt_mean_vehicle/4)
        edge = False
        edge = rectangle_close_to_edge(x, y, w, h, upper_limit, lower_limit, right_limit, left_limit)
        if edge == False:
            #Create crop to use as input for the CNN
            crop_path = crop_cnn_image(gray, x,y,w,h)
            #Check if box is a car and if so append to list
            vehicle = generate_predictions(crop_path, sess2,input_tensor, end_points)
            if  vehicle == True:
                boxes_in_frame.append((x, y, w, h))
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)# top-left corner and bottom-right corner of rectangle
        else:
            pass
    
    return boxes_in_frame


def rectangle_inside_another(new_rectangle, existing_object):
    '''Check if new_rectangle is inside existing_object'''
    [leftA, topA, wa, ha] = new_rectangle
    [leftB, topB, wb, hb] = existing_object
    rightA, bottomA = leftA + wa, topA + ha
    rightB, bottomB = leftB + wb, topB + hb
    if( topA <= topB ) or ( bottomA >= bottomB ) or ( leftA <= leftB ) or ( rightA >= rightB ):
        return False
    elif ( topA == topB ) or ( bottomA == bottomB ) or ( rightA == rightB ) or ( leftA == leftB ):
        return False
    else:
        return True

def find_intersection(box, existing_box, overlap_threshold, box_found):
    '''This function finds out if there is an intersection between two objects'''
    [XA1, YA1, wa, ha] = existing_box
    [XB1, YB1, wb, hb] = box
    XA2 = XA1 + wa
    XB2 = XB1 + wb
    YA2 = YA1 + ha
    YB2 = YB1 + hb
    SI= np.max([0, np.min([XA2, XB2]) - np.max([XA1, XB1])]) * np.max([0, np.min([YA2, YB2]) - np.max([YA1, YB1])])
    if SI>0:
        SA = rect_area(XA1, XA2, YA1, YA2)
        SB = rect_area(XB1, XB2, YB1, YB2)
        SU = SA + SB - SI
        overlap = SI / SU
        if overlap > overlap_threshold:
            box_found = True
            return box_found
        else:
            return box_found
        
def find_new_objects(existing_objects, boxes_in_frame, frame, overlap_threshold, cars_counter, object_ids):
    ''' This function checks for new objects in the frame by looking at the intersection between the new objects detected and the
        already identified as objects '''
    new_objects = []
    boxes_exists = True
    for box in boxes_in_frame:
        box_found = False
        if existing_objects == []:
            boxes_exists = False
        if boxes_exists == True:
            i = 0
            for existing_box in existing_objects:
                inside_rect = rectangle_inside_another(box, existing_box)
                if inside_rect ==True:
                    existing_objects[i] = box
                else:
                    box_found = find_intersection(box, existing_box, overlap_threshold, box_found)
                    
                i+=1
            
            if box_found == False:
                cv2.putText(frame, "Status: {}".format('NOT FOUND'), (box[0], box[2]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                existing_objects.append(box)
                new_objects.append(box)
                cars_counter +=1
                object_ids.append(cars_counter)
                
        else:
            existing_objects.append(box)
            new_objects.append(box)
            cars_counter +=1
            object_ids.append(cars_counter)
        
    return new_objects, existing_objects, cars_counter, object_ids



def get_vehicle_size(background, sizes, gray, min_area, frame, initial):
    '''This function returns a list where each item corresponds to the coordinates of a box where openCV detects pixel motion'''
    if initial ==  True:
        background = gray
    else:
        #threshold is 25 --> If the delta is less than 25, we discard the pixel and set it to black 
        print('ok:', background.size, gray.size)
        frameDelta = cv2.absdiff(background, gray)
        background = gray
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        im2, cnts, hierarchy= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                sizes.append(cv2.contourArea(c))
                continue    
    return sizes
 
def object_detection(gray, existing_objects, initial, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold, cars_counter, object_ids, sizes, sess2, input_tensor, end_points):
    '''
        This function analyse the frame checks it the object detected does not exist in previous frames.
        First frame --> use as background
        Second frame --> use to identify initial objects
        @initial defines if it is the first iteration '''
    
    if initial == True:
        background = gray
        new_objects = find_boxes_in_frame(background, gray, upper_limit, lower_limit, right_limit, left_limit, min_area, frame, sizes, sess2, input_tensor, end_points)
        existing_objects = new_objects
        boxes_in_frame = []
        #cars_counter = len(new_objects)
    else:
        #Get background image to compare the frame with
        background = cv2.imread(frame_paths[i-delay+1],)
        background = cv2.resize(background,(299, 299), interpolation = cv2.INTER_CUBIC)
        gray1 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        background = gray1
        
        boxes_in_frame = find_boxes_in_frame(background, gray, upper_limit, lower_limit, right_limit, left_limit, min_area, frame, sizes, sess2, input_tensor, end_points)
        
        new_objects, existing_objects, cars_counter, object_ids = find_new_objects(existing_objects, boxes_in_frame, frame, overlap_threshold, cars_counter, object_ids)
            
    
    return new_objects, existing_objects, boxes_in_frame, cars_counter, object_ids

def create_trackers(new_item, trackers, frame):
    trackers.append(cv2.Tracker_create("MIL"))
    ok = trackers[-1].init(frame, new_item)
    ok, bb = trackers[-1].update(frame)
    return trackers, ok

def rectangle_close_to_edge(x, y, w, h, upper_limit, lower_limit, right_limit, left_limit):
    '''This function identifies if a rectangle that is being tracked is too close to the edge of the frame. (0,0) is in the top left corner'''
    x1 = x + w
    y1 = y + h
    if (y < upper_limit) or  (y1>lower_limit) or  (x<left_limit) or  (x1>right_limit):
        #print("rectangle_close_to_edge (width, height):" ,(x1-x) ,(y1-y))
        #wait = input("PRESS ENTER TO CONTINUE.")
        return True        
    else:
        return False

def crop_image(img, x,y,w,h, count):
    #Increment the size of the crop
    threshold = 10
    x, y, w, h = x - threshold, y - threshold, w + threshold, h + threshold
    crop_img = img[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
    path = "./crops_test/" + str(count) +'.jpg'
    cv2.imwrite(path, crop_img)  
    return path

def crop_cnn_image(img, x,y,w,h):
    #Increment the size of the crop for the cnn case
    threshold = 10
    x, y, w, h = x - threshold, y - threshold, w + threshold, h + threshold
    crop_img = img[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    path = "./crops_test/" + str(date_string) +'.jpg'
    cv2.imwrite(path, crop_img)  
    return path


def track_frames(trackers, background, gray, existing_objects, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold, cars_counter, object_ids, sizes, sess2, input_tensor, end_points):
    ''' Track the frames '''
    new_objects = []
    boxes_in_frame = []
    ok = True
    if background is None:
        initial = True
        new_objects, existing_objects, boxes_in_frame, cars_counter, object_ids = object_detection(gray, existing_objects, initial, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold, cars_counter, object_ids, sizes, sess2, input_tensor, end_points)
        initial = False 
        for objeto in existing_objects:
            #Create a bbox element with those coordinates
            trackers, ok = create_trackers(objeto, trackers, gray)
        background = 'not None'
    elif i>delay: #@delay is included so that we don't start checking for pixel modification after a certain number of frames
        #Start looking for new objects due to there is enough time difference to identify some pixel movement
        initial = False 
        new_objects, existing_objects, boxes_in_frame, cars_counter, object_ids = object_detection(gray, existing_objects, initial, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold, cars_counter, object_ids, sizes, sess2, input_tensor, end_points)
        # returns [(231, 162, 311, 250), (150, 140, 243, 228), (304, 87, 343, 141)]
        for objeto in new_objects:
            #Create a bbox element with those coordinates
            trackers, ok = create_trackers(objeto, trackers, gray)
    
    return new_objects, existing_objects, boxes_in_frame, trackers, background, ok,  cars_counter, object_ids

def video_to_frames(videoPath, name):
    '''Video to frames'''
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 0
    success = True
    frame_paths = []
    while success:
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        if success == True:
            frame_name = "./frames/%s_frameID_%d.jpg" % (name, count)
            frame_paths.append(frame_name)
            #print(name)
            cv2.imwrite(frame_name, image)     # save frame as JPEG file
            count += 1
          
    return frame_paths

def update_tracker(rccn_new_objects, new_objects, existing_objects, boxes_in_frame, trackers, i, gray, frame, name, frame_id, object_ids):
    '''Update tracker'''
    j=0
    bboxes = []
    for tracker_item in trackers:
        ok, bb = trackers[j].update(gray)
        (x1, y1, w1, h1) = bb
        '''CROP IMAGE'''
        if frame_id % 8== 0:
            counter_crop = 'folder_file_' + name +'_'+ '_frameID_' +  str(i) + '_objID_' + str(object_ids[j]) +'_coordinates_'+ str(int(x1))  +'_'+  str(int(y1)) +'_'+ str(int(w1)) +'_'+ str(int(h1))  
            crop_image(frame, x1,y1,w1,h1, counter_crop)
            
        #if exists a rectangle inside this one then reduce the size
        for object in boxes_in_frame:
            inside_rect = rectangle_inside_another(object, bb)
            if inside_rect ==True:
                #replace tracker with new window size
                #object = tuple(0.9*x for x in object) 
                trackers[j] = cv2.Tracker_create("MIL")
                ok = trackers[j].init(gray, bb)
                ok, bb = trackers[j].update(gray)
                bb = object
        for object in rccn_new_objects:
            inside_rect = rectangle_inside_another(object, bb)
            if inside_rect ==True:
                trackers[j] = cv2.Tracker_create("MIL")
                ok = trackers[j].init(gray, bb)
                ok, bb = trackers[j].update(gray)
                bb = object
        existing_objects[j] = bb
        bboxes.append(bb)
        j+=1
            
    return bboxes, existing_objects, trackers, object_ids

def delete_boxes_and_crop_vehicles(existing_objects, crop_count, i, frame, upper_limit, lower_limit, right_limit, left_limit, bboxes, trackers, name, object_ids):
    k=0
    for existing_box in existing_objects:
        edge = False
        (x, y, w, h) = existing_box
        edge = rectangle_close_to_edge(x, y, w, h, upper_limit, lower_limit, right_limit, left_limit)
        if edge:
            del existing_objects[k]
            del trackers[k]
            del bboxes[k]
            del object_ids[k]
        k+=1
    
    return existing_objects, trackers, bboxes, object_ids
    
def get_and_preProcess_frame(frame_paths, i):
    '''Identify new frames'''
    #GET FRAME
    frame = cv2.imread(frame_paths[i],)
    # resize the frame, convert it to grayscale, and blur it
    frame = cv2.resize(frame,(299, 299), interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray

def draw_bounding_box(frame, bboxes, ok):
        ''' Draw bounding box '''
        for bb in bboxes:
            if ok:
                p1 = (int(bb[0]), int(bb[1]))
                p2 = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
                cv2.rectangle(frame, p1, p2, (0,0,255))
        # Display result
        cv2.imshow("Tracking", frame)

def adjust_bbox_values(rcnn_boxes):
    adjusted_rcnn_boxes = []
    for rcnn_box in rcnn_boxes:
        rcnn_box_values = [rcnn_box[0], rcnn_box[1], rcnn_box[2] - rcnn_box[0], rcnn_box[3] - rcnn_box[1]]
        adjusted_rcnn_boxes.append(rcnn_box_values)
    return adjusted_rcnn_boxes

