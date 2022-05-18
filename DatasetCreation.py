#%% Importing packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

#%% Defining Functions
#############################################################
#############################################################

def load_image_names(folder):
    
    file_list = []
    for file_name in os.listdir(folder):
        if file_name[0] != '.':
            file_list.append(file_name)

    return(file_list)


#%% Reading the contents of the dataset directory

# Current directory is on separate hard drive
dataset_directory = '/media/briancottle/Samsung_T5/ML Dataset'
file_names = load_image_names(dataset_directory)
