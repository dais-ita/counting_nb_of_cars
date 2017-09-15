#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


# # --------------------------------------------------------
# # Tensorflow CNN --> Vehicle or Empty road prediction
# # --------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
# import vgg_preprocessing
# from vgg import vgg_16, vgg_arg_scope

import os
import time

# ================ DATASET INFORMATION ======================

import os
import tensorflow as tf;
import numpy as np
from PIL import Image
from inception_resnet_v2 import inception_resnet_v2 as slim_irv2  # PYHTONPATH should contain the slim/ directory in the tensorflow/models repo.
# ATOL = 1e-5
# VERBOSE = True

def vis_detections(im, class_name, dets, thresh=0.1, rcnn_boxes):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #inds = np.where(dets[:, -1])[0]
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds: #more than one box per image
        bbox = dets[i, :4]
        rcnn_boxes.append(bbox)
        score = dets[i, -1]
        #bbox contain the values we are after
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    return rcnn_boxes


def demo(sess1, net, image_name):
    CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

    """Detect object classes in an image using pre-computed object proposals."""
    rcnn_boxes = []
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    print(image_name)
    im = cv2.imread(image_name)
    print('imageName', image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess1, net1, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.3 #Non-max suppression is a way to eliminate points that do not lie in important edges.
    NMS_THRESH = 0.01
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #print('CLASE:', cls, cls_ind)
        cls_ind += 1# because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if cls == 'car':
            rcnn_boxes = vis_detections(im, cls, dets, thresh=CONF_THRESH, rcnn_boxes)
            print('There are car classes', dets)

    return rcnn_boxes

def parse_args():
    DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
    NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


def define_cnn_session():
    """
    Code modified from here: [https://github.com/tensorflow/models/issues/429]
    """
    checkpoint_file = 'tf_int/model.ckpt-25933'
    SLIM_CKPT = checkpoint_file
    g_1 = tf.Graph()
    with g_1.as_default():
        # Setup preprocessing
        input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

    # Setup session
    sess2 = tf.Session(graph=g_1)
    arg_scope = inception_resnet_v2_arg_scope

    # Load the model
    with g_1.as_default():
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(scaled_input_tensor, num_classes=2, is_training=False)
        variables_to_restore = slim.get_variables_to_restore()
        saver2 = tf.train.Saver(variables_to_restore)
    saver2.restore(sess2, SLIM_CKPT)

    return sess2, input_tensor, end_points


def generate_predictions(image, sess2, input_tensor, end_points):
    dic_cars = {'1': 'car', '0': 'empty_road'}
    pred = 'test'
    value = 0
    im = Image.open(image).resize((299, 299))
    arr = np.expand_dims(np.array(im), axis=0)
    y_pred = sess2.run([end_points['Predictions']], feed_dict={input_tensor: arr})
    print('y_pred', y_pred)
    y_pred = y_pred[0].ravel()
    print('y_pred1:', y_pred)
    max = np.max(y_pred)
    if max > 0.6:
        pred = np.argmax(y_pred)
        value = y_pred[pred]
    else:
        pred = np.argmin(y_pred)
        value = y_pred[pred]

    value1 = dic_cars[str(pred)]

    print("{} class={} prob={}".format(image, value1, value))
    if value1 == 'car':
        return True
    else:
        return False

def imgs2png(filename, dest_folder):
    im = Image.open(filename)
    name = filename.split('/')[-1].split('.')[0]
    root_folder = '/'.join(filename.split('/')[:-1]) + '/'
    new_path = root_folder + dest_folder + '/' + name + '.png'
    print('New path png', new_path)
    im.save(new_path)
    print('images converted to .png format')
    return new_path

def runFasterRCNN(filename_png, sess1, net1):
    import glob
    from PIL import Image
    # im_names = []
    # for filename in file_names:
    #     print('filename', filename)
    #     if filename != []:
    #         for file in filename:
    #             im_names.append(file)
    # print ('im_names', im_names)
    # print('Loaded network {:s}'.format(tfmodel))
    # for im_name in im_names:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    rcnn_boxes = demo(sess1, net1, filename_png)
    print('BOXES', rcnn_boxes)
    #plt.show()
    return rcnn_boxes

def defineFasterRCNN():
    DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
    NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    cwd = os.getcwd()
    print(cwd)
    tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    print('tfModel PATH:',tfmodel)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess1 = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net1 = vgg16(batch_size=1)
    elif demonet == 'res101':
        net1 = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net1.create_architecture(sess1, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])

    saver1 = tf.train.Saver()
    saver1.restore(sess1, tfmodel)

    return sess1, net1
















