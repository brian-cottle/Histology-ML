# %% importing packages

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from skimage import measure
import cv2 as cv

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
    one_hot_seg = tf.one_hot(tf.squeeze(segmentation-1),4,axis=-1)

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
    dataset = dataset.shuffle(1000)

    # adding the batch size to the dataset
    dataset = dataset.batch(batch_size=batch_size)

    return(dataset)

#############################################################
#############################################################
# %%
# pick one directory from which to read the dataset shards
shard_dataset_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards'
os.chdir(shard_dataset_directory)
file_names = tf.io.gfile.glob(shard_dataset_directory + "/shard_*_of_*.tfrecords")

# retrieve a dataset including all of the files in the directory. To only 
# include information from a training dataset, you should move specific dataset
# files to a new directory, and then from there perform the analysis.
dataset = get_dataset(file_names,batch_size=1)


# %%
percentages = []

# iterate through each example in the dataset
for sample in dataset:
    ground_truth = sample[1]
    # sum up each class in the dataset for this example
    sum0 = np.sum(ground_truth[0,:,:,0])
    sum1 = np.sum(ground_truth[0,:,:,1])
    sum2 = np.sum(ground_truth[0,:,:,2])
    sum3 = np.sum(ground_truth[0,:,:,3])
    sum4 = np.sum(ground_truth[0,:,:,4])

    # append the sums to the list
    percentages.append([sum0,sum1,sum2,sum3,sum4])
    gc.collect()
    tf.keras.backend.clear_session()

# %%

# convert to array
sums = np.asarray(percentages)
# get the sum of each class divided by the number of pixels in an image. make 
# sure to change this value if your images are a different size!
percents = sums/(1024*1024)

# get the standard deviation of the percentages
std_dev = np.std(percents,axis=0)
# get the mean of the percentages
means = np.mean(percents,axis=0)
# produce the weights as the inverse of the percentages
weights = 1/means

# put the current weights calculation here for future reference so you don't 
# have to run this code if you forget it. It takes a while...
# weights = array([28.78661087,  3.60830475,  1.63037567, 14.44688883])

# %%
