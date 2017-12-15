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

This project turned out to be a great learning exercise, but did go as planned in terms of timeline.  This project lead to what felt like several small projects or segments.  In the end, we used PyTorch to delivere a convolutional neural network model with some preprocessing.  While I mentioned the use of Keras above, I chose to go with PyTorch because I wanted experience using a GPU.  

The stack:

* Python3.5
* Ubuntu16.04
* PyTorch
* TensorFlow
* Jupyter Notebook

If the project were to have been broken up correctly it would be as follows:

A. Image classification research
B. Installing a second hard drive for a Linux distro
C. Installing and configuring Ubunutu with Cuda support
D. Installing Python/Tensorflow/PyTorch 
E. Learning and writing image processing techniques
F. Learning Convolutions and Convolutional Neural Networks
G. Data exploration and image manipulation
H. Exploring neural networks with Pytorch
I. Testing pre-processing
J. Model Selection

#### Exploration

The data:

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/Data.png)

Original Images:

Icebergs -

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/IcebergB1B2.png )

Ships - 

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/ShipsB1B2.png )

A deeper look - in 3D:

Iceberg -

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/Iceberg3d.png )

Ship - 

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/Ship3d.png )


From here I had an understanding of the data and how to look at images, so I went to the next logical step of transforming what I had into something a little bit more clear.  This is where I started playing with convolutions - passing arrays over my bands to alter the data in a way that would help highlight the focus of the pictures.
Below are a few of the gradients:

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/Gradients.png)

One example of these is an edge detection gradient -

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/EdgeDetection.png)

After a while, I then tried mixing them together or layering them.  This is mostly in attempt to learn how these work, as they are also in the neural network model that we get to below.

Modified Images:

The left two images are ships and the right two are icebergs -

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/MultipleConv.png)

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/MultipleConv2.png)

#### Convolutional Neural Network

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/ModelSummary.png)

#### Results

Below is an example of an output from the model testing against a validation set.

![alt text](https://github.com/DSNeville/Practicum-II-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/Images/ValidationLoss.png)

The model performs reasonably, but not that great compared to the competition leaderboard.
Conceptually there were many very intersting methods that I did not get a chance to try.  Transforming the images and making predictions takes quite a bit of time on these larger data files.  Realistically, if I had more powerful hardware, things might go more smoothly as I would be able to iterate more quickly through alterations.  I am using an Nvidia 970, and there are many upgrades that would improve performance on the market.
I would have liked to try methods that look into color filters and more established techniques.  In this I was building gradients and kernels just based off my understanding, but as the project went on, my understanding continued to grow.  This would have been a much different project if I were to start it now.
Reflecting on PyTorch - I found it rather complicated at first, but got used to it by the end.  The one thing that was lacking was documentation.  It is harder to learn with less examples and in this case, PyTorch seems to be less utilized than other neural network frameworks.  
