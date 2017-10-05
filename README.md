# Counting Number of cars

This project contains three main modules:

	1- Faster RCNN  module is executed every certain number of frames and a number of vehicles are identified (the first frame is analysed by this module).
	2- Optical flow is executed and a number of boxes which may contain a vehicle are identified. This is run every certain amount of frames but much smaller than in the case of the Faster RCNN module
	3- CNN module checks the boxes identified by the optical flow module and decides if the box contains a vehicle or not. This is executed for every box identified by the optical flow module.


The result of this project is the amount of cars that appear in each video.

# Summary

Pull docker image from:

https://hub.docker.com/r/squintana/dl-docker/

Download the full project from the following location:

	https://drive.google.com/open?id=0B4RgtXiS2li0QksxaUFpRWxBaEE

Place the downloaded project in the docker sharedfolder (<local_folder>).

Run docker image using the following command:

nvidia-docker run -it -p 8888:8888 -v ~/<local_folder>:/root/sharedfolder squintana/dl-docker bash

Run ~/sharedfolder/tools/main.py to obtain the results.
	
Videos are stored in tf-faster-rcnn-master/tools/videos_test1


### Requirements
1. Python 2.7.x
2. TensorFlow == 1.0.1 
3. OpenCV 3.3
4. Cuda

