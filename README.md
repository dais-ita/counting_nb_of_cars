# Counting Number of cars

This project contains three main modules:

	1- Faster RCNN  module is executed every certain number of frames and a number of vehicles are identified (the first frame is analysed by this module).
	2- Optical flow is executed and a number of boxes which may contain a vehicle are identified. This is run every certain amount of frames but much smaller than in the case of the Faster RCNN module
	3- CNN module checks the boxes identified by the optical flow module and decides if the box contains a vehicle or not. This is executed for every box identified by the optical flow module.


The result of this project is the amount of cars that appear in each video.

# Summary

Download the full project from the following location:

	https://drive.google.com/open?id=0B4RgtXiS2li0dGVVVUdXLXA2dE0

Run main.py to obtain the results.
	
Videos are stored in tf-faster-rcnn-master/tools/videos_test1


### Requirements
1. Python 2.7.x
2. TensorFlow >= 0.12
3. OpenCV 2.4.2


## ResNet model setup instructions:

are:

* You need first to pull the [TensorFlow models repo] (https://github.com/tensorflow/models) into your disk:
```
git clone https://github.com/tensorflow/models.git
```
* Then, compile the proto files there:
```
cd models
protoc object_detection/protos/*.proto --python_out=.
```


* Finally, update the "car_counting_demo.py" line 26 with the correct path of the tensorflow models repo:
```
TF_MODELS_REPO_PATH = '/home/malzantot/Nesl/models'
```

* To run the demo on a given image:

```
python car_counting_demo.py --image_path=test_image.jpg 
```
The setup instructions:

are:

* You need first to pull the [TensorFlow models repo] (https://github.com/tensorflow/models) into your disk:
```
git clone https://github.com/tensorflow/models.git
```
* Then, compile the proto files there:
```
cd models
protoc object_detection/protos/*.proto --python_out=.
```


* Finally, update the "car_counting_demo.py" line 26 with the correct path of the tensorflow models repo:
```
TF_MODELS_REPO_PATH = '/home/malzantot/Nesl/models'
```

* To run the demo on a given image:

```
python car_counting_demo.py --image_path=test_image.jpg 
```

