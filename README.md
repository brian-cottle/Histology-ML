# Histology-ML


## About this repository
This repository contains the code that I have been using for my research at the University of Utah. Currently, this repository is limited to the code used for generating the dataset used directly for semantic segmentation, and training a custom U-Net network for semantic segmentation from those data. Quite a bit of other code has been written in MATLAB that performs image registration and creates 3D models from the segmentations provided. The preparation and migration of the code to GitHub is taking a while, though, so stay tuned for that code to be available.

## about this project
Our research focuses on improving open-heart surgery for pre-term infants. In short, massive effort, training, technology, and experience are leveraged in order to provide functioning hearts infants who are born with congenitally deformed hearts. Unfortunately, despite the extraordinary technology available and experience of the surgeon's, crucial structures within the heart (specifically called the Sinus node, AV node and their related structures) which help it beat properly are often damaged during these surgeries. This means that these newborn patients will need a permanent pacemaker implanted to keep their heart beating properly for the rest of their life, imposing an entirely new set of challenges and complications associated with maintaining these devices.

Think about it, open heart surgery on an adult heart is an extraordinarily complex procedure, and those surgeries are performed within an open heart which is roughly the size of your fist. When operating on congenitally deformed hearts in premature infants, the patient's hearts are no bigger than a small strawberry. To add to the complexity, many of the critical and important parts of the heart are out of place from where they normally are. Yikes!

Damage to the cardiac conduction system, which includes the sinus node, AV node, and a few other important structures, is what makes it so that patients need a permanent pacemaker implanted after their surgeries in order to make sure their hearts beat correctly. These parts of the heart get damaged during surgery because, in these congenitally deformed hearts, the location of the nodes under the surface of the heart isn't visible to the surgeons. On top of that their location can vary without any indication. Considering that you need to have a precision of sometimes less than a millimeter to make sure you place your stitches correctly in the heart, knowing exactly where the node is so that you can work confidently in dangerous areas in the heart is critical.

Our research focuses on providing surgeons with the knowledge and confidence they need in order to avoid damaging the sinus and AV nodes during surgery. We do this with two projects we are working on:

1. Using histology, machine learning, and image processing we are creating high-resolution, data-rich models of the nodal regions to provide precise visualizations that aren't available to surgeon's yet. This project is where our this repository comes into play.

2. Using custom-developed probes, light scattering spectroscopy, and machine learning we are developing a method for using optical biopsy technology for intraoperative tissue characterization. Essentially, we are developing a tool that a surgeon can use to inspect the tissue much deeper into the heart than current technologies (millimeters instead of micrometers). 

## About the data
The data that we work with on this project involves thousands of large full-slide histological images from donor tissue samples of the cardiac conduction system in human infants. These images are of tissues stained using an automated Masson's Trichrome method, and imaged using an automated slide scanner (Zeiss Axioscan). We have data from roughly 12 sinus nodes and 12 AV nodes, which amounts to roughly 8,000 images totaling around 900GB of image data.  

## About the network
The network used in this work is close to a vanilla U-Net network. Now, I know that everyone and their dog has produced some riff on the original U-Net, so why would I end up choosing something so close to the original? Well, in our experimentation, the original *works*, and it works quite well. Sure, we added some batch normalization and a few other more recent tweaks, but in the end, why would we choose something with significantly larger computational overhead for a limited gain in performance? One of the main reasons we believe the U-Net performs so well is because of the training dataset we have generated. 

To train the network, we have a dataset of sparsely segmented images containing our tissues of interest (vasculature, neural, connective, and muscular tissues). The original U-Net was created to be trained on relatively small datasets (often between 100 and 500 images total) specifically for image segmentation in biological datasets, and it works well in that regard. We, using our sparse segmentation methods, have a dataset with more than 10,000 unique images for training alone, which gives our network a very wide base to train on. Additionally, because of the consistency of coloring and staining provided by our staining and imaging methods, the images are very consistent and require relatively little processing before presentation to the network. 

## About the files
The process of generating the dataset involves a pipeline following these scripts:

1. DatasetCreation.py: this file takes the provided images from a given directory and creates the sparse dataset for training the network. This takes random tiles from each image that contain the poorly represented classes (vasculature and neural tissues) and saves them as separate images in a different directory.

2. TF_WriteToRecordFile.py: This script takes the tiled, sparse dataset and saves it into a sharded TFRecord dataset. This was necessary for a variety of reasons. One major reason was that when loading these images from a directory, given their size (roughly 1500px by 1500px), the training would take something close to a couple of days for only a handful of epochs. Using the TFRecords enabled us to reduce this training time from the order of days per epoch to about 5-10 minutes per epoch. 

3. uNet*.py: These files all have purposes for training and using the network. The uNet_Functional.py and uNet_Subclassed.py files contain the same network, I just wanted to prove to myself that I could create the same complicated network using both the functional API and sub-classing a network. the *FullImage_Segmentation.py and *FullImage_Segmentation_FullDirectory.py scripts segment a full image and an entire directory of images, respectively. 

Other files provide support but are not necessarily needed to use the network. As this project is still in the development phase, files will come and go, and significant changes to the project structure will be implemented over time.

## About the contributions
While I have worked with numerous individuals, research groups, and institutions for this project, all of the code for this aspect of the project (generation of 3D models from histological images) up to this point has been writen by me. As other undergraduate students complete their smaller, senior projects involved in this work their code will be uploaded and attributed.   

