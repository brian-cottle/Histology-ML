# %% importing packages

import argparse
from typing import NamedTuple
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import tqdm
from natsort import natsorted
from glob import glob
import magic
import re
from PIL import Image

plt.rcParams['figure.figsize'] = [50, 150]



# %% Citations
#############################################################
#############################################################

# You should try and make this a "select this directory and pad/process all the 
# files in it this way" script. You currently have WAY too much repetition
# %%
TissueChainID = '26/01/'
cropping = 0

old_files = 0

fiduciary_files = 1
nodal_files = 1
seg_files = 1
high_res_files = 1

nodal_white = 1
fiduciary_white = 1

reduction_size = 4

if reduction_size == 4:
    reduction_name = 'QuarterScale'

if reduction_size == 8:
    reduction_name = 'EighthScale'



base_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID
JPG_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'JPG/'
jpg_file_names = glob(JPG_directory + '*.jpg')


if old_files:
    ML_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'uNet_Segmentations/'
    Nodal_Seg_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'Segmentations/Nodal Segmentation FullScale_NoPad/'
    if fiduciary_files:
        Fiduciary_Seg_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'Segmentations/Fiduciary Segmentation FullScale_NoPad/'
else:
    ML_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'uNet_Segmentations/'
    Nodal_Seg_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'Segmentations/Nodal Segmentation/'
    if fiduciary_files:
        Fiduciary_Seg_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'Segmentations/Fiduciary Segmentation/'

high_res_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'HighResSeg/'

os.chdir(ML_directory)
jpg_file_names = natsorted(jpg_file_names)
padding_size = 4000 # + 1536

# %% USE THIS SECTION FOR CROPPING THE SEGMENTATIONS AFTER THE UNET HAS AT IT
if cropping:
    for idx in tqdm.tqdm(range(len(jpg_file_names))):

        out_directory = './../Cropped_uNet_Segmentations/'
        
        # create the directory for saving if it doesn't already exist
        if not os.path.isdir(out_directory):
            os.mkdir(out_directory)

        os.chdir(out_directory)

        jpg_file = jpg_file_names[idx]
        id = jpg_file.split('/')[-1].split('.')[0]
        id = id.split('_')[0] + '_' + id.split('_')[1] + '_' + id.split('_')[2]
        ml_file = glob(ML_directory + f'{id}_*.png')[0]

        jpg_image1 = cv.imread(jpg_file)
        ml_image1 = cv.imread(ml_file)[:,:,0]
        [x,y,z] = jpg_image1.shape

        cropped_ml1 = ml_image1[padding_size:padding_size+x,
                                padding_size:padding_size+y]

        cv.imwrite(
            id + 
            f'_CroppedSeg.png',
            cropped_ml1
            )

# %%
ML_directory = '/var/confocaldata/HumanNodal/HeartData/'+ TissueChainID +'Cropped_uNet_Segmentations/'
ml_file_names = glob(ML_directory + '*.png')
all_image_sizes = []
for file_name in ml_file_names:
    header = magic.from_file(file_name)
    size = re.search('(\d+) x (\d+)',header).groups()
    sizes = [int(a) for a in size]
    all_image_sizes.append(sizes)

max_width = np.max(np.asarray(all_image_sizes)[:,0])
max_height = np.max(np.asarray(all_image_sizes)[:,1])

idx = 200
additional_padding = 4000
os.chdir(JPG_directory)
out_big_directory = base_directory + 'Padded_Images/'
out_small_directory = base_directory + 'Padded_Images_' + reduction_name + '/'
out_parent_list = [out_big_directory,out_small_directory]
out_list = []
if not os.path.isdir(out_big_directory):
    os.mkdir(out_big_directory)

if not os.path.isdir(out_small_directory):
    os.mkdir(out_small_directory)

for idx, out_directory in enumerate(out_parent_list):
    jpg_out = out_directory + 'JPG'
    seg_out = out_directory + 'Seg'
    nodal_out = out_directory + 'Nodal'
    fiduciary_out = out_directory + 'Fiduciary'
    high_res_out = out_directory + 'HighRes'
    out_list.append([jpg_out,seg_out,nodal_out,high_res_out,fiduciary_out])

    for directory in out_list[idx]:
        if not os.path.isdir(directory):
            os.mkdir(directory)



for idx in tqdm.tqdm(range(len(jpg_file_names))):

    jpg_file = jpg_file_names[idx]
    id = jpg_file.split('/')[-1].split('.')[0]
    id = id.split('_')[0] + '_' + id.split('_')[1] + '_' + id.split('_')[2]
    
    
    
    # Create a separate section for the nodal tissue stuff, as it looks like 
    # nodal segmentation will actually happen after the registration using the
    # segmentations
    # change this to _*.png if you are not using the FullScale_NoPad segmentations
    # will need to scale the segmentations for newer segmentations that haven't been 
    # performed using previously padded images. This section is below, starting with


    jpg_image = cv.imread(jpg_file)
    [height,width,z] = jpg_image.shape

    height_diff = max_height - height
    width_diff = max_width - width

    if height_diff%2 == 1:
        pad_top = np.floor(height_diff/2) + additional_padding
        pad_bottom = np.floor(height_diff/2) + additional_padding
        pad_bottom += 1
    else:
        pad_top = np.floor(height_diff/2) + additional_padding
        pad_bottom = np.floor(height_diff/2) + additional_padding

    if width_diff%2 == 1:
        pad_left = np.floor(width_diff/2) + additional_padding
        pad_right = np.floor(width_diff/2) + additional_padding
        pad_right += 1
    else:
        pad_left = np.floor(width_diff/2) + additional_padding
        pad_right = np.floor(width_diff/2) + additional_padding

    padded_jpg = cv.copyMakeBorder(jpg_image,
                                int(pad_top),
                                int(pad_bottom),
                                int(pad_left),
                                int(pad_right),
                                borderType=cv.cv2.BORDER_CONSTANT,
                                value=[255,255,255])

    os.chdir(out_list[0][0])
    cv.imwrite(
        id + 
        f'_Padded.png',
        padded_jpg
        )

    [pad_height,pad_width,z] = padded_jpg.shape
    

    width_small = int(pad_width/reduction_size)
    height_small = int(pad_height/reduction_size)
    jpg_small = cv.resize(padded_jpg,[width_small,height_small],cv.INTER_AREA)
    
    os.chdir(out_list[1][0])
    cv.imwrite(
        id + 
        f'_Padded_' + reduction_name + '.png',
        jpg_small
        )


    if seg_files:
        ml_file = glob(ML_directory + f'{id}_*.png')[0]
        ml_image = cv.imread(ml_file)[:,:,0]
        padded_seg = cv.copyMakeBorder(ml_image,
                                    int(pad_top),
                                    int(pad_bottom),
                                    int(pad_left),
                                    int(pad_right),
                                    borderType=cv.cv2.BORDER_CONSTANT,
                                    value=[0,0,0])
        os.chdir(out_list[0][1])
        cv.imwrite(
            id + 
            f'_Padded_Seg.png',
            padded_seg
            )

        seg_small = np.array(Image.fromarray(padded_seg).resize((width_small,height_small), Image.NEAREST))
        os.chdir(out_list[1][1])
        cv.imwrite(
            id + 
            f'_Padded_Seg_' + reduction_name + '.png',
            seg_small
            )


    if nodal_files:

        if old_files:
            nodal_file = glob(Nodal_Seg_directory + f'{id}-*.png')[0]
        else:
            nodal_file = glob(Nodal_Seg_directory + f'{id}_*.png')[0]

        nodal_image = cv.imread(nodal_file)[:,:,0]
        # be warry of this, you may need to use this later, though I'm not sure what
        # it was originally used for.
        if nodal_white:
            if sum(sum(nodal_image)) > 0:
                nodal_image = ~nodal_image
        # This accounts for the nodal segmentation images being a quarter the
        # original size, but you should make sure that you haven't already done the 
        # fullscale noPad stuff yet
        if ~old_files:
            nodal_image = np.array(Image.fromarray(nodal_image).resize((width,height), Image.NEAREST))

        padded_nodal = cv.copyMakeBorder(nodal_image,
                                    int(pad_top),
                                    int(pad_bottom),
                                    int(pad_left),
                                    int(pad_right),
                                    borderType=cv.cv2.BORDER_CONSTANT,
                                    value=[0,0,0])

        os.chdir(out_list[0][2])
        cv.imwrite(
            id + 
            f'_Padded_Nodal.png',
            padded_nodal
            )

        nodal_small = np.array(Image.fromarray(padded_nodal).resize((width_small,height_small), Image.NEAREST))
        os.chdir(out_list[1][2])
        cv.imwrite(
            id + 
            f'_Padded_Nodal_' + reduction_name + '.png',
            nodal_small
            )





    if high_res_files:
        high_res_file = glob(high_res_directory + f'{id}_*.png')[0]
        high_res_image = cv.imread(high_res_file)[:,:,0]

        padded_high_res = cv.copyMakeBorder(high_res_image,
                                    int(pad_top),
                                    int(pad_bottom),
                                    int(pad_left),
                                    int(pad_right),
                                    borderType=cv.cv2.BORDER_CONSTANT,
                                    value=[0,0,0])


        os.chdir(out_list[0][3])
        cv.imwrite(
            id + 
            f'_Padded_HighRes.png',
            padded_high_res
            )

        high_res_small = np.array(Image.fromarray(padded_high_res).resize((width_small,height_small), Image.NEAREST))
        os.chdir(out_list[1][3])
        cv.imwrite(
            id + 
            f'_Padded_HighRes_' + reduction_name + '.png',
            high_res_small
            )





    if fiduciary_files:
        if old_files:
            fiduciary_file = glob(Fiduciary_Seg_directory + f'{id}-*.png')[0]
        else:
            fiduciary_file = glob(Fiduciary_Seg_directory + f'{id}_*.png')[0]

        fiduciary_image = cv.imread(fiduciary_file)[:,:,0]

        if fiduciary_white:
            if sum(sum(fiduciary_image)) > 0:
                fiduciary_image = ~fiduciary_image

        if ~old_files:
            fiduciary_image = np.array(Image.fromarray(fiduciary_image).resize((width,height), Image.NEAREST))

        padded_fiduciary = cv.copyMakeBorder(fiduciary_image,
                                    int(pad_top),
                                    int(pad_bottom),
                                    int(pad_left),
                                    int(pad_right),
                                    borderType=cv.cv2.BORDER_CONSTANT,
                                    value=[0,0,0])
        os.chdir(out_list[0][4])
        cv.imwrite(
            id + 
            f'_Padded_Fiduciary.png',
            padded_fiduciary
            )

        fiduciary_small = np.array(Image.fromarray(padded_fiduciary).resize((width_small,height_small), Image.NEAREST))
        os.chdir(out_list[1][4])
        cv.imwrite(
            id + 
            f'_Padded_Fiduciary_' + reduction_name + '.png',
            fiduciary_small
            )

 # %%
