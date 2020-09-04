# Face Liveness Detection
---
## Description:
A deep-learning pipeline capable of spotting fake vs legitimate faces and performing anti-face spoofing in face recognition systems. It is built with the help of Keras, Tensorflow, and OpenCV. A sample dataset is uploaded in the sample_dataset_folder.

## Method
The problem of detecting fake faces vs real/legitimate faces is treated as a binary classification task. Basically, given an input image, we’ll train a Convolutional Neural Network capable of distinguishing real faces from fake/spoofed faces. There are 4 main steps involved in the task:
 1. Build the image dataset itself.
 2. Implement a CNN capable of performing liveness detector(Livenessnet).
 3. Train the liveness detector network.
 4. Create a Python + OpenCV script capable of taking our trained liveness detector model and apply it to real-time video.
 5. Create a webplatform to access the liveness detection algorithm in an interactive manner.

## Working flow
![alt text](https://github.com/sakethbachu/liveness_detection/blob/master/sample_liveness_data/Desc%20info/workflow.png "Logo Title Text 1")

## Demo
![alt text](https://github.com/sakethbachu/liveness_detection/blob/master/sample_liveness_data/Desc%20info/liveness.jpeg "Logo Title Text 1")
