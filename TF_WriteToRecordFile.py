# %% importing packages

import numpy as np
import tensorflow as tf
from skimage import measure
import cv2 as cv
import os
import tqdm
import matplotlib.pyplot as plt

# %% Citations
#############################################################
#############################################################

# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
# https://keras.io/examples/keras_recipes/creating_tfrecords/
# https://www.tensorflow.org/tutorials/load_data/tfrecord

# %% Defining TF Records Helper Functions
#############################################################
#############################################################

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_image(full_image_seg,bbox):

    data = {
        'height' : int64_feature(full_image_seg.shape[0]),
        'width' : int64_feature(full_image_seg.shape[1]),
        'raw_image' : bytes_feature(
            tf.io.serialize_tensor(full_image_seg[:,:,0:3])),
        'raw_seg' : bytes_feature(
            tf.io.serialize_tensor(full_image_seg[:,:,3])),
        'bbox_x' : float_feature_list(bbox[0]),
        'bbox_y' : float_feature_list(bbox[1]),
        'bbox_width' : float_feature_list(bbox[2]),
        'bbox_height' : float_feature_list(bbox[3])
    }

    parsed_image = tf.train.Example(features=tf.train.Features(feature=data))

    return(parsed_image)

# Defining Functions
#############################################################
#############################################################

def get_bounding_boxes(binary_image):
    # This function receives a binary image, and returns a list of the
    # bounding boxes that surround the positive connected components

    labeled_image = measure.label(binary_image) # labeling image
    regions = measure.regionprops(labeled_image) # getting region props
    
    box_x = []
    box_y = []
    box_width = []
    box_height = []

    # iterating over the number of regions found in the image
    for region in regions:
        # retrieving [min_row, min_col, max_row, max_col]
        bounding_box = region['bbox'] 

        # Calculating the center of the bounding box, as this is a common
        # format for the box parameters in SSD networks
        x = np.floor(np.mean([bounding_box[2],bounding_box[0]]))
        y = np.floor(np.mean([bounding_box[3],bounding_box[1]]))

        # retriving the width and height, also common format
        width = bounding_box[2]-bounding_box[0]
        height = bounding_box[3]-bounding_box[1]

        # appending to list for storage
        box_x.append(x)
        box_y.append(y)
        box_width.append(width)
        box_height.append(height)


    return([box_x,box_y,box_width,box_height])

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

def parse_image_bbox(file_name,bbox_class_id):
    image = cv.imread(file_name,cv.IMREAD_UNCHANGED)
    color_image = image[:,:,0:3]
    seg = image[:,:,3]
    bbox_seg = seg == bbox_class_id
    bbox = get_bounding_boxes(bbox_seg)
    parsed_image = parse_image(image,bbox)

    return(parsed_image)

#############################################################

def get_shard_sizes(file_names,max_files_per_shard):
    num_images = len(file_names)
    num_splits = num_images//max_files_per_shard + 1
    if num_images%max_files_per_shard == 0:
        num_splits -= 1
    
    print(f'Using {num_splits} shards to store {num_images} images.')

    return(num_splits,max_files_per_shard)

#############################################################

def write_all_images_to_shards(file_names,
                               num_splits,
                               max_files_per_shard,
                               bbox_id=5):

    out_directory = './../dataset_shards/'
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)
    
    files_written = 0

    for idx in tqdm.tqdm(range(num_splits)):
        current_shard_name = out_directory + \
            f'shard_{idx+1}_of_{num_splits}.tfrecords'
        
        writer = tf.io.TFRecordWriter(current_shard_name)
        
        num_files_this_shard = 0

        while num_files_this_shard < max_files_per_shard:
            image_idx = idx*max_files_per_shard + num_files_this_shard
            if image_idx == len(file_names):
                break
            parsed_image = parse_image_bbox(file_names[image_idx],bbox_id)

            writer.write(parsed_image.SerializeToString())

            num_files_this_shard += 1
            files_written += 1

        writer.close()
    
    print(f'{files_written} files have been written to the tfrecord.')
    return(files_written)

#############################################################

def parse_tf_elements(element):

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

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    raw_seg = content['raw_seg']
    raw_image = content['raw_image']
    bbox_x = content['bbox_x']
    bbox_y = content['bbox_y']
    bbox_height = content['bbox_height']
    bbox_width = content['bbox_width']

    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    image = tf.reshape(image,shape=[height,width,3])
    segmentation = tf.io.parse_tensor(raw_seg, out_type=tf.uint8)
    segmentation = tf.reshape(segmentation,shape=[height,width,1])

    # bbox = [bbox_x,bbox_y,bbox_height,bbox_width]

    return(image,segmentation) # 

#############################################################
#############################################################

# %%

dataset_directory = '/media/briancottle/Samsung_T5/sub_sampled_20220531'
os.chdir(dataset_directory)
file_names = load_image_names(dataset_directory)
num_splits,max_files_per_shard = get_shard_sizes(file_names,500)

write_all_images_to_shards(file_names,num_splits,max_files_per_shard,bbox_id=5)

# %%
os.chdir('/media/briancottle/Samsung_T5/dataset_shards')
dataset = tf.data.TFRecordDataset('shard_5_of_21.tfrecords')
dataset = dataset.map(parse_tf_elements)
# %%
for sample in dataset.take(2):
    plt.imshow(sample[0])
    plt.show()
    plt.imshow(sample[1])
    plt.show()
# %%

B = dataset.take(30)
# %%
