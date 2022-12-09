# %% importing packages

import numpy as np
from skimage import measure
from skimage import morphology
from skimage import segmentation
from scipy import ndimage
import cv2 as cv
import os
import matplotlib.pyplot as plt
import tqdm
import random
from glob import glob
from natsort import natsorted
from fpdf import FPDF
plt.rcParams['figure.figsize'] = [50, 150]


# %% Sources
#############################################################
#############################################################

# https://pypi.org/project/img2pdf/
# https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
# https://stackoverflow.com/questions/27327513/create-pdf-from-a-list-of-images
# https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

# Defining Functions
#############################################################
#############################################################

def filter_fun_vasculature(file_name:str):
    '''I realize this isn't very functional programming, but this function
    is paired with the "filter_fun_nerves" function for filtering file names
    according to the class for which the tile was selected using the 
    DatasetCreation.py file'''
    class_id = 'vasc_seg'
    
    if class_id in file_name:
        return(True)
    else:
        return(False)

#############################################################

def filter_fun_nerves(file_name:str):
    '''Performs filtering of file names according to the class for which the
    tile was selected when using the DatasetCreation.py file'''
    class_id = 'neural_seg'
    
    if class_id in file_name:
        return(True)
    else:
        return(False)


#############################################################


def sample_image_names_from_directory(directory,number_of_samples):
    '''Randomly selects number_of_samples file names of each of the two 
    different classes (vasculature and neural) from the sub_sampled
    dataset created by the DatasetCreation.py script. Returns those file 
    names.'''

    os.chdir(directory)    
    file_names = glob('./*.png')
    
    vasc_filt = filter(filter_fun_vasculature,file_names)
    nerve_filt = filter(filter_fun_nerves,file_names)

    vasc_image_names = list(vasc_filt)
    nerve_image_names = list(nerve_filt)

    rand_vasc_images = random.choices(vasc_image_names,k=number_of_samples)
    rand_nerve_images = random.choices(nerve_image_names,k=number_of_samples)

    random_image_names = rand_vasc_images + rand_nerve_images

    random.shuffle(random_image_names)

    return(random_image_names)

#############################################################

def create_segment_outline_image(image_name):
    '''Uses openCV to create an border for the segmentations of interest in the 
    image whose file name is provided as an argument. Returns an image with the 
    vasculature highlighted in red, and the neural tissue highlighted in 
    green.'''

    current_image = cv.imread(image_name,cv.IMREAD_UNCHANGED)
    color = current_image[:,:,0:3]
    # color = cv.cvtColor(color,cv.COLOR_BGR2RGB)
    seg = current_image[:,:,3]
    dilation_amount = 50
    vasculature = seg==5
    neural = seg==6

    vasculature = morphology.binary_dilation(
        vasculature,
        np.ones((dilation_amount,dilation_amount))
        )

    neural = morphology.binary_dilation(
        neural,
        np.ones((dilation_amount,dilation_amount))
        )

    vasc_contours,_heirarchy = cv.findContours(vasculature.astype(np.uint8),
                                               cv.RETR_EXTERNAL,
                                               cv.CHAIN_APPROX_NONE)
    
    neural_contours,_heirarchy = cv.findContours(neural.astype(np.uint8),
                                                 cv.RETR_EXTERNAL,
                                                 cv.CHAIN_APPROX_NONE)

    contoured_image = cv.drawContours(
        np.ascontiguousarray(color,np.uint8), vasc_contours, -1, (0,0,255), 10
        )
    contoured_image = cv.drawContours(
        contoured_image, neural_contours, -1, (0,255,0), 10
        )
    

    return(contoured_image)

#############################################################

def save_images_to_PDF(image_name_list,file_name,create_contour = True):
    '''Uses the above functions to save images sequentially to a PDF for use
    in validating the segmentations performed. The PDF has space at the bottom
    for adding check boxes in a separate program such as Adobe Acrobat.'''

    pdf = FPDF()
    pdf.set_auto_page_break(False)
    pdf.set_left_margin(margin=5)

    save_directory = dataset_directory + '/../PDF_images/'

    current_directory = os.getcwd()

    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)


    for idx in tqdm.tqdm(range(len(image_name_list))):
        if create_contour:
            image_name = image_name_list[idx]

            contoured_image = create_segment_outline_image(image_name)
            
            image_name_split = image_name.split('/')[1].split('.')[0]
            new_image_name = image_name_split+'_outlined.png'

            os.chdir(save_directory)
            cv.imwrite(new_image_name,contoured_image)

            pdf.add_page(format=(610,350))
            pdf.image(new_image_name,h=300)
            os.chdir(current_directory)
        else:
            image_name = image_name_list[idx]
            pdf.add_page(format=(610,350))
            pdf.image(image_name,h=300)

    os.chdir(save_directory)

    pdf.output(file_name,'F')

    return(True)

#############################################################
#############################################################
# %%

dataset_directory = '/var/confocaldata/HumanNodal/HeartData/08/01/PDFImages'

filtered_names = sample_image_names_from_directory(dataset_directory,20)

save_images_to_PDF(filtered_names,'test_file.pdf',create_contour=False)

# %%
