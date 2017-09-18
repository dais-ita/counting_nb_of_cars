"""
Using Faster-RCNN to count cars based on TensorFlow object detection model.

Author: Moustafa Alzantot (malzantot@ucla.edu)
Date: 17/09/2017
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import glob
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf

import argparse

## Important: We need to checkout the TensorFlow models repo:
## https://github.com/tensorflow/models
## and add its path to the sys.path environmental variable variable.
TF_MODELS_REPO_PATH = '/home/malzantot/Nesl/models'
MSCOCO_LABELS_PATH = TF_MODELS_REPO_PATH + '/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

sys.path.append(TF_MODELS_REPO_PATH)
sys.path.append(TF_MODELS_REPO_PATH+'/object_detection')


from utils import label_map_util
from utils import visualization_utils as vis_util

def maybe_download_model(model_name):
    MODEL_FILENAME = model_name + '.tar.gz'
    MODEL_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.exists(model_name):
        print('Downloading model ', model_name)
        # Download the model
        opener = urllib.request.URLopener()
        opener.retrieve(MODEL_DOWNLOAD_BASE  + MODEL_FILENAME, MODEL_FILENAME)
        tar_file = tarfile.open(MODEL_FILENAME)
        
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
                
def create_detection_graph(ckpt_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
        
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        type=str, required = True,
                        help='Path for input image')
    args = parser.parse_args()
    
    MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    maybe_download_model(MODEL_NAME)
    detection_graph = create_detection_graph(PATH_TO_CKPT)
    
    label_map = label_map_util.load_labelmap(MSCOCO_LABELS_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image = Image.open(args.image_path)
            image_np = load_image_into_numpy_array(image)
            # shape it as [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # Perform actual detection
            (boxes_, scores_, classes_, num_detections_) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, np.squeeze(boxes_),
                np.squeeze(classes_).astype(np.int32),
                np.squeeze(scores_),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            #plt.imshow(image_np)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image_np)
            plt.show()

    
    
    
        
    
