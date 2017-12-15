# Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge
Project for Regis University Practicum II Class 


This project explore the challenge hosted on Kaggle:
https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

Why this project?
  I have done very little in image processing, but see it as a great way to diversify my data science portfolio.
  The applications seem to be endless once you get the hang of it.
  
### Initial Proposal

John P Neville
jpneville@gmail.com	
(248) 703-8915
Iceberg Classifier Challenge

Description:

The Iceberg Classifier Challenge is a competition hosted on Kaggle for classifying if a picture is an iceberg or a ship.  
The pictures are scans from a radar type of system that come from a satellite, but are not the actual photo.  In the scans, 
objects are represented as white dots in a black background of water.

The objective is to provide an estimate for each photo that represents the likelihood that the object is an iceberg.  
This number can be between 1 and 0.  Submissions are evaluated based on the log loss between our predictions and the real values.

Analysis:

For this project I will be using the python environment to discover which pictures contain icebergs.  I have done one other 
project where I used photos, but it was the digit recognizer, and I used R to complete the analysis.  I have seen several 
kernels on Kaggle, with many explorations into how to analyze photos.  I will likely need to do some research before diving into
my actual approach here, but some places to start would be to get familiar with Keras, a deep learning library that can deal with 
images.  Keras runs over Tensorflow, another deep learning tool that has methods for dealing with image classification.

The size of this data set is rather large for a school project, so I would like to also experiment in using a GPU to process this data.  For that segment, Pytorch seem like a good place to start.
In terms of classification and analysis, I am not certain where I will end up, but I know that if I could utilize logistic 
regression, it would get me the type of “answer” that I am looking for.  Other than that, I will need to explore which types 
of algorithms work for image classification.  

Data:

The data is in a json format, and about 300 MB.  It contains a test ant train set, hosted on Kaggle.

It is described in the competition:

	Id – Iceberg ID
	Band_1, & Band_2  - 75x75 pixel values in a list
	Inc_angle – The incidence angle that the picture was taken at
	Is_iceberg – 1 = Yes, 0 = No

Timeline:

Week 2 – Research and Data PreProcessing
Week 3 – Explore Data in Multiple Formats (EDA)
Week 4-7 – Apply Research and Evaluate Models
Week 8 – Prepare Presentation and Submit Results

### Project Retrospective

This project turned out to be a great learning exercise, but did go as planned in terms of timeline.  This project lead to what felt like several small projects or segments.  In the end, we used PyTorch to delivered a convolutional neural network model with some preprocessing.  While I mentioned the use of Keras above, I chose to go with PyTorch because I wanted experience using a GPU.  

The stack:

Python3.5
Ubuntu16.4
PyTorch
TensorFlow
Jupyter Notebook

If the project were to have been broken up correctly it would be as follows:

A.  Image classification research
B.  Installing a second hard drive for a Linux distro
C.  Installing and configuring Ubunutu with Cuda support
D.  Installing Python/Tensorflow/PyTorch 
E.  Learning and writing image processing techniques
F.  Learning Convolutions and Convolutional Neural Networks
G.  Data exploration and image manipulation
H.  Exploring neural networks with Pytorch
I.  Testing pre-processing
J.  Model Selection

#### Exploration

The data:

![alt text]( Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/Images/Data.png )

Original Images:

Icebergs -

![alt text]( Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/Images/IcebergB1B2.png )

Ships - 

![alt text]( Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/Images/ShipsB1B2.png )

A deeper look - in 3D:

Iceberg -

![alt text]( Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/Images/Iceberg3d.png )

Ship - 

![alt text]( Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/Images/Ship3d.png )



Modified Images:

#### Convolutional Neural Network

#### Results

