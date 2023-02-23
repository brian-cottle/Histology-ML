# %% importing packages

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage import measure
import cv2 as cv
import time
from joblib import Parallel, delayed
from scipy.stats import variation 
import tqdm
import matplotlib.pyplot as plt
import gc



# %% Citations
#############################################################
#############################################################


def parse_tf_elements(element):
    '''This function is the mapper function for retrieving examples from the
       tfrecord'''

    # create placeholders for all the features in each example
    data = {
        'height' : tf.io.FixedLenFeature([],tf.int64),
        'width' : tf.io.FixedLenFeature([],tf.int64),
        'raw_image' : tf.io.FixedLenFeature([],tf.string),
        'raw_seg' : tf.io.FixedLenFeature([],tf.string),
        'bbox_x' : tf.io.VarLenFeature(tf.float32),
        'bbox_y' : tf.io.VarLenFeature(tf.float32),
        'bbox_height' : tf.io.VarLenFeature(tf.float32),
        'bbox_width' : tf.io.VarLenFeature(tf.float32)
    }

    # pull out the current example
    content = tf.io.parse_single_example(element, data)

    # pull out each feature from the example 
    height = content['height']
    width = content['width']
    raw_seg = content['raw_seg']
    raw_image = content['raw_image']
    bbox_x = content['bbox_x']
    bbox_y = content['bbox_y']
    bbox_height = content['bbox_height']
    bbox_width = content['bbox_width']

    # convert the images to uint8, and reshape them accordingly
    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    image = tf.reshape(image,shape=[height,width,3])
    segmentation = tf.io.parse_tensor(raw_seg, out_type=tf.uint8)
    segmentation = tf.reshape(segmentation,shape=[height,width,1])
    one_hot_seg = tf.one_hot(tf.squeeze(segmentation),7,axis=-1)

    # there currently is a bug with returning the bbox, but isn't necessary
    # to fix for creating the initial uNet for segmentation exploration
    
    # bbox = [bbox_x,bbox_y,bbox_height,bbox_width]

    return(image,one_hot_seg)

#############################################################

def load_dataset(file_names):
    '''Receives a list of file names from a folder that contains tfrecord files
       compiled previously. Takes these names and creates a tensorflow dataset
       from them.'''

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(file_names)

    # you can shard the dataset if you like to reduce the size when necessary
    # dataset = dataset.shard(num_shards=2,index=1)
    
    # order in the file names doesn't really matter, so ignoring it
    dataset = dataset.with_options(ignore_order)

    # mapping the dataset using the parse_tf_elements function defined earlier
    dataset = dataset.map(parse_tf_elements,num_parallel_calls=1)
    
    return(dataset)

#############################################################

def get_dataset(file_names,batch_size):
    '''Receives a list of file names of tfrecord shards from a dataset as well
       as a batch size for the dataset.'''
    
    # uses the load_dataset function to retrieve the files and put them into a 
    # dataset.
    dataset = load_dataset(file_names)
    
    # creates a shuffle buffer of 1000. Number was arbitrarily chosen, feel free
    # to alter as fits your hardware.
    dataset = dataset.shuffle(50)

    # adding the batch size to the dataset
    dataset = dataset.batch(batch_size=batch_size)

    return(dataset)

#############################################################

def joblib_parallel_function_class_sums(sample):
    '''Receives a sample, separates out the ground truth, and sends back 
       a list of the sums of each class in order.'''
    
    # sum up each class in the dataset for this example
    sum_classes = np.sum(sample,axis=(0,1,2))
    try:
        return(sum_classes)
    finally:
        gc.collect()
        tf.keras.backend.clear_session()

#############################################################
# %%
# pick one directory from which to read the dataset shards
shard_dataset_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_6/train'
os.chdir(shard_dataset_directory)
file_names = tf.io.gfile.glob(shard_dataset_directory + "/shard_*_of_*.tfrecords")

# retrieve a dataset including all of the files in the directory. To only 
# include information from a training dataset, you should move specific dataset
# files to a new directory, and then from there perform the analysis.
batch_size = 1
dataset = get_dataset(file_names,batch_size=batch_size)
dataset = dataset.shard(num_shards=10,index=2)
# %%

# percentages = Parallel(
#     n_jobs=15, verbose=5, backend='loky')(delayed(joblib_parallel_function_class_sums)
#     (sample[1]) for sample in dataset
#     )

# %% iterate through each example in the dataset
percentages = []
variances = np.zeros((32400,3))
all_means = np.zeros((32400,3))
count = 0
for sample in dataset:
    ground_truth = sample[1]
    image = sample[0]
    # sum up each class in the dataset for this example
    sum_classes = np.sum(ground_truth,axis=(0,1,2))

    # all_means[count,:] = [np.mean(image[0,:,:,0]),
    #                     np.mean(image[0,:,:,1]),
    #                     np.mean(image[0,:,:,2])]
    
    # variances[count,:] = [np.var(image[0,:,:,0]),
    #                   np.var(image[0,:,:,1]),
    #                   np.var(image[0,:,:,2])]

    # append the sums to the list
    percentages.append(sum_classes)
    gc.collect()
    tf.keras.backend.clear_session()
    count += 1

# %%

# convert to array
sums = np.asarray(percentages)
# get the sum of each class divided by the number of pixels in an image. make 
# sure to change this value if your images are a different size!
percents = sums/(1024*1024*batch_size)

# get the standard deviation of the percentages
std_dev = np.std(percents,axis=0)
# get the mean of the percentages
means = np.mean(percents,axis=0)
# produce the weights as the inverse of the percentages
weights = 1/means
print(weights)

# mean_means = np.mean(np.asarray(all_means),axis=0)
# var_vars = np.var(np.asarray(variances),axis=0)
# print(f'mean of the means is {mean_means}')
# print(f'variance of the variances is {var_vars}')




# [         inf   2.15248481   3.28798466   5.18559616  46.96594578
#  130.77512742 105.23678672]


# [         inf   2.3971094    3.04084893   4.77029963  39.23478673
#  118.13505703  96.22377396]



# [         inf   2.72403952   2.81034368   4.36437716  36.66264202
#  108.40694198  87.39903838] # results with dataset 5

# means and variances with daataset 5

# mean of the means is [232.69476802 204.16933591 211.45184799]
# variance of the variances is [ 139869.85259648  550311.88980989 1160687.94506812]


# %%
