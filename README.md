# Face Liveness Detection
![alt text](https://github.com/sakethbachu/Face-Liveness-Detection/blob/master/sample_liveness_data/Desc%20info/livenessg.gif "Logo Title Text 1")

---
## Description
A deep-learning pipeline capable of spotting fake vs legitimate faces and performing anti-face spoofing in face recognition systems. It is built with the help of Keras, Tensorflow, and OpenCV. A sample dataset is uploaded in the sample_dataset_folder.

## Method
The problem of detecting fake faces vs real/legitimate faces is treated as a binary classification task. Basically, given an input image, we’ll train a Convolutional Neural Network capable of distinguishing real faces from fake/spoofed faces. There are 4 main steps involved in the task:
 1. Build the image dataset itself.
 2. Implement a CNN capable of performing liveness detector(Livenessnet).
 3. Train the liveness detector network.
 4. Create a Python + OpenCV script capable of taking our trained liveness detector model and apply it to real-time video.
 5. Create a webplatform to access the liveness detection algorithm in an interactive manner.

## Contents of this repository
1. sample_liveness_data : contains the sample dataset.
2. Face Liveness Detection -Saketh.pptx : A couple of slides that will give you information on th project and our motivation.
3. demo.py : Our demonstration script will fire up your webcam to grab frames to conduct face liveness detection in real-time.
4. deploy.prototxt : Support file for pretrained face detector. 
5. le.pickle : Our class label encoder.
6. liveness.model : The liveness model file.
7. livenessnet.py : The python file containing the model.
8. res10_300x300_ssd_iter_140000.caffemodel: Pretrained face detector.
9. train_liveness.py: The python script to train the model.


## Working flow
![alt text](https://github.com/sakethbachu/liveness_detection/blob/master/sample_liveness_data/Desc%20info/workflow.png "Logo Title Text 1")

## Further work
1. Gathering data having a larger set of ethnicity and different types of fake/spoofed photos.
2. Adding more heuristics to team up with deep-learning.
