# %% Importing packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import patches
plt.rcParams['figure.figsize'] = [5, 10]
import os
import tensorflow as tf
from scipy.spatial import distance
import time
from glob import glob
from joblib import Parallel, delayed
from natsort import natsorted
plt.rcParams['figure.figsize'] = [50, 150]

# %% Citations
#############################################################
#############################################################



# %% Defining Functions
#############################################################
#############################################################

def process_file(file,
                 original_seg_directory,
                 high_res_seg_directory,
                 save_directory):

    try:
        file_id = file.split('-')[0] + '_'
        high_res_file = glob(high_res_seg_directory + file_id + '*.png')
        assert len(high_res_file) > 0, \
            f'could not find the pair to {file}, skipping file!'

        original_image = cv.imread(file,cv.IMREAD_UNCHANGED)
        original_seg = original_image[:,:,3]
        high_res_seg = cv.imread(high_res_file[0],cv.IMREAD_UNCHANGED)
        # 1: background, 2: muscle nuclei, 3: connective nuclei
        # 4: muscle, 5: connective, 6: blood

        original_seg[high_res_seg==1] = 0
        original_seg[high_res_seg==6] = 7

        original_image[:,:,3] = original_seg

        new_name = file_id + 'ML5.png'

        os.chdir(save_directory)
        cv.imwrite(new_name,original_image)
        os.chdir(original_seg_directory)


    except AssertionError as e:
        print(e)


#############################################################
#############################################################

original_seg_directory = '/media/briancottle/Samsung_T5/ML_Dataset_3'
high_res_seg_directory = '/home/briancottle/Research/Semantic_Segmentation/'\
                        'High_Res_Ilastik_Segmentation_V10/'\
                        'Ilastik_Segmentation_V10/'

save_directory = '/media/briancottle/Samsung_T5/ML_Dataset_5'

if not os.path.isdir(save_directory):
    os.mkdir(save_directory)

os.chdir(original_seg_directory)
original_files = glob('*.png')

contains_names_vascular = Parallel(
    n_jobs=22, verbose=5)(delayed(process_file)
    (file = name,
    original_seg_directory = original_seg_directory,
    high_res_seg_directory = high_res_seg_directory,
    save_directory = save_directory) for name in original_files
    )
