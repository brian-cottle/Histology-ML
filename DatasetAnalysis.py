# %% Importing packages

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed

# %% Defining Functions
#############################################################
#############################################################

def load_image_names(folder):
    # This function reads the images within a folder while filtering 
    # out the weird invisible files that macos includes in their folders
    
    file_list = []
    for file_name in os.listdir(folder):

        # check if the first character of the name is a '.', skip if so
        if file_name[0] != '.': 
            file_list.append(file_name)

    return(file_list)

#############################################################

def basic_image_seg_visualization(image_name):
    # this function shows both the RGB image and its corresponding segmentation
    # next to each other
    image = cv.imread(image_name,cv.IMREAD_UNCHANGED)
    # change the color order, because openCV2 reads color in as BGR, not RGB
    color_image = cv.cvtColor(image[:,:,0:3],cv.COLOR_BGR2RGB)

    # create our subplots
    fig, (ax1, ax2) = plt.subplots(1,2)

    # show both the images
    ax1.imshow(color_image)
    ax2.imshow(image[:,:,3], vmin=0,vmax=5)
    plt.show()
    return()

#############################################################

def includes_segmentation(image_name, class_id):
    # this function is meant to be run in parallel, but can be run individually.
    # it receives an image name and a class_id, and determins whether and how 
    # much of a certain class is contained in the image.

    image = cv.imread(image_name,cv.IMREAD_UNCHANGED)
    segmentation = image[:,:,3]

    # how many pixels are a part of the class?
    seg_sum = np.sum((segmentation==class_id))
    # total number of pixels?
    total_sum = segmentation.shape[0]*segmentation.shape[1]

    # if the class is in the image, return true and the percentage
    if seg_sum>0:
        return(image_name,True,seg_sum/total_sum)
    else:
        return(image_name,False,0)

#############################################################
#############################################################

dataset_directory = '/media/briancottle/3a7b7bdc-6753-4423-b5ac-ff074ad75013/sub_sampled_20220526'

os.chdir(dataset_directory)

file_names = load_image_names('.')

# %%
random_index = int(np.random.random()*len(file_names))
basic_image_seg_visualization(file_names[random_index])

# %%
class_id = 5

# check for vasculature
contains_names_vascular = Parallel(n_jobs=20, verbose=1)(delayed(includes_segmentation) \
                                    (name,5) for name in file_names)
# check for neural tissue 
contains_names_neural = Parallel(n_jobs=20, verbose=1)(delayed(includes_segmentation) \
                                    (name,4) for name in file_names)


# %%

# get only the names and percentages for images that contain vasculature
vascular_images = []
vascular_percentages = []
for evaluation in contains_names_vascular:
    name = evaluation[0]
    present = evaluation[1]
    percentage = evaluation[2]

    if present:
        vascular_images.append(name)
        vascular_percentages.append(percentage)

# get only the names and percentages for images that contain vasculature
neural_images = []
neural_percentages = []
for evaluation in contains_names_neural:
    name = evaluation[0]
    present = evaluation[1]
    percentage = evaluation[2]

    if present:
        neural_images.append(name)
        neural_percentages.append(percentage)

# %% 


# reporting on valuse found for vasculature
print(f'the number of images containing vasculature is: {len(vascular_images)}')
print(f'the largest percentage was found in file {vascular_images[np.argmax(vascular_percentages)]}, and was {np.max(vascular_percentages)}')
basic_image_seg_visualization(vascular_images[np.argmax(vascular_percentages)])

plt.hist(vascular_percentages)
plt.show()
# %%
random_index = int(np.random.random()*len(vascular_images))
basic_image_seg_visualization(vascular_images[random_index])
# %%
