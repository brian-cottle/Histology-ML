# %% importing packages

import numpy as np
import tensorflow as tf
from skimage import measure
import skimage.transform as transform
import cv2 as cv
import os
import tqdm
import matplotlib.pyplot as plt
import random

# %% Citations
#############################################################
#############################################################

# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
# https://keras.io/examples/keras_recipes/creating_tfrecords/
# https://www.tensorflow.org/tutorials/load_data/tfrecord

# %% Defining TF Records Helper Functions
#############################################################
#############################################################

# These following functions were created from the TDS blog and 
# the tensorflow suggestions/tutorial

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_image(full_image,seg,bbox,image_name):

    data = {
        'height' : int64_feature(full_image.shape[0]),
        'width' : int64_feature(full_image.shape[1]),
        'raw_image' : bytes_feature(
            tf.io.serialize_tensor(full_image)),
        'raw_seg' : bytes_feature(
            tf.io.serialize_tensor(seg)),
        'bbox_x' : float_feature_list(bbox[0]),
        'bbox_y' : float_feature_list(bbox[1]),
        'bbox_width' : float_feature_list(bbox[2]),
        'bbox_height' : float_feature_list(bbox[3]),
        'name' : bytes_feature(bytes(image_name, encoding='utf-8')),
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
    
    # lists for storing the sequence of x, y, width, and height data for 
    # the image. They are separate so that each can be its own list that 
    # is stored, seemed to make sense with the limitations of the 
    # float_feature_list thing. 
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

        # appending to respective lists for storage
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

def parse_image_bbox(file_name,bbox_class_id,reduction_size=1):
    # This function receives a name and the class id of which you want to 
    # provide bounding boxes for in the dataset
    image = cv.imread(file_name,cv.IMREAD_UNCHANGED)
    passed = False
    parsed_image = 0
    if image.shape[0] == image.shape[1]:
        passed = True
        # separating out the color image
        try:
            color_image = image[:,:,0:3]
        except Exception as e:
            print(file_name)
            print(image.shape)
            print(e)
            
        # downsampling the color image
        if reduction_size > 1:
            height = color_image.shape[0]
            width = color_image.shape[1]

            height2 = int(height/reduction_size)
            width2 = int(width/reduction_size)

            color_image = cv.resize(color_image,[height2,width2],cv.INTER_AREA)

        # getting the segmentation for the bbox production, making compensations
        # for odd segmentations
        seg = image[:,:,3]
        # seg[seg==4] = 2
        # seg[seg==5] = 4
        # seg[seg==6] = 5
        # seg[seg==7] = 6

        if reduction_size > 1:
            height = seg.shape[0]
            width = seg.shape[1]

            height2 = int(height/reduction_size)
            width2 = int(width/reduction_size)

            seg = cv.resize(seg,[height2,width2],cv.INTER_NEAREST)

        # creating the binary image for bboxes
        bbox_seg = seg == bbox_class_id
        bbox = get_bounding_boxes(bbox_seg)

        parsed_image = parse_image(color_image,seg,bbox,image_name=file_name)

    return(parsed_image,passed)
    

#############################################################

def get_shard_sizes(file_names,max_files_per_shard):
    # This function receives a list of file names and the maximum number of 
    # files you want to save in a shard.
    num_images = len(file_names)
    # determining the number of splits for the image. The +1 collects any 
    # stragglers that don't completely "fill up" a shard. It is removed if the 
    # number comes out with no remainder. 
    num_splits = num_images//max_files_per_shard + 1
    if num_images%max_files_per_shard == 0:
        num_splits -= 1
    
    print(f'Using {num_splits} shards to store {num_images} images.')

    return(num_splits,max_files_per_shard)

#############################################################

def write_all_images_to_shards(file_names,
                               num_splits,
                               max_files_per_shard,
                               bbox_id=5,
                               reduction_size=1):
    # this function receives a list of file names, the number of splits
    # produced by get_shard_sizes, the how many files will be put in each
    # shard, and the class id of which segmentation you want to produce
    # bounding boxes for. In the future you could easily add the functionality
    # to produce bboxes for both vasculature and neural tissues.
    out_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_6/'

    # create the directory for saving if it doesn't already exist
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    # keeps track of how many files have been written total    
    files_written = 0

    # displaying progress based on number of shards successfully saved
    for idx in tqdm.tqdm(range(num_splits)):
    
        current_shard_name = out_directory + \
            f'shard_{idx+1}_of_{num_splits}.tfrecords'
        
        # create the writer for the current shard
        writer = tf.io.TFRecordWriter(current_shard_name)
        
        # keep track of how many files we've put in this shard
        num_files_this_shard = 0

        # exit if we hit the max number of files for this shard
        while num_files_this_shard < max_files_per_shard:
            # keeping track of names of files across shards
            image_idx = idx*max_files_per_shard + num_files_this_shard
            # if we hit the end of all the files, stop this shard
            if image_idx == len(file_names):
                break
            # get a parsed image
            parsed_image,passed = parse_image_bbox(file_names[image_idx],
                                            bbox_id,
                                            reduction_size=reduction_size)

            # add the current image to the tfrecord file
            if passed:
                writer.write(parsed_image.SerializeToString())

                num_files_this_shard += 1
                files_written += 1
            else:
                print(f'{file_names[image_idx]} failed, skipping')
                num_files_this_shard += 1

        writer.close()
    
    print(f'{files_written} files have been written to the tfrecord.')
    return(files_written)

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
        'bbox_width' : tf.io.VarLenFeature(tf.float32),
        'name' : tf.io.FixedLenFeature([],tf.string),
    }

    # pull out the current example
    content = tf.io.parse_single_example(element, data)

    # pull out each feature from the example 
    height = content['height']
    width = content['width']
    raw_seg = content['raw_seg']
    raw_image = content['raw_image']
    name = content['name']
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

    return(image,one_hot_seg,name)

#############################################################
#############################################################

# %%
# writing the files to a new directory!
dataset_directory = '/home/briancottle/Research/Semantic_Segmentation/sub_sampled_20221129'
os.chdir(dataset_directory)
file_names = load_image_names(dataset_directory)
num_splits,max_files_per_shard = get_shard_sizes(file_names,400)

# %%
write_all_images_to_shards(file_names,
                           num_splits,
                           max_files_per_shard,
                           bbox_id=5,
                           reduction_size=1)

# %% loading an example shard, and creating the mapped dataset
os.chdir('/home/briancottle/Research/Semantic_Segmentation/dataset_shards_6/')
dataset = tf.data.TFRecordDataset('shard_10_of_102.tfrecords')
dataset = dataset.map(parse_tf_elements)
# %% 
# double checking some of the examples to make sure it all worked well!
for sample in dataset.take(100):
    print(sample[2])
    plt.imshow(cv.cvtColor(np.asarray(sample[0]),cv.COLOR_BGR2RGB))
    print(sample[0].shape)
    plt.show()
    seg = tf.argmax(sample[1],axis=-1)
    plt.imshow(seg,vmin=0,vmax=6)
    plt.show()
    print(np.max(seg))
    print(np.unique(seg))
    


# %% Checking for dud/empty files

for file_name in file_names:
    image = cv.imread(file_name,cv.IMREAD_UNCHANGED)
    try:
        print(image.shape)
        assert image.shape == (1024,1024,3)
    except Exception as e:
        print(e)
        print(file_name)
    


# %% Further data exploration to ensure proper storage in the record files!
GT = []
Images = []
interested = 12
count = 0
for sample in dataset.take(30):
    if count == interested:
        GT.append(sample[1])
        gt = tf.squeeze(sample[1])
        Images.append(sample[0])
        # print('layer 7')
        # plt.imshow(gt[:,:,7])
        # print(np.sum(gt[:,:,7]))
        # plt.show()
        print('layer 6')
        plt.imshow(gt[:,:,6])
        print(np.sum(gt[:,:,6]))
        plt.show()
        print('layer 5')
        plt.imshow(gt[:,:,5])
        print(np.sum(gt[:,:,5]))
        plt.show()
        print('layer 4')
        plt.imshow(gt[:,:,4])
        print(np.sum(gt[:,:,4]))
        plt.show()
        print('layer 3')
        plt.imshow(gt[:,:,3])
        print(np.sum(gt[:,:,3]))
        plt.show()
        print('layer 2')
        plt.imshow(gt[:,:,2])
        print(np.sum(gt[:,:,2]))
        plt.show()
        print('layer 1')
        plt.imshow(gt[:,:,1])
        print(np.sum(gt[:,:,1]))
        plt.show()
        print('layer 0')
        plt.imshow(gt[:,:,0])
        print(np.sum(gt[:,:,0]))
        plt.show()
    count += 1

# %%
# directory where the dataset shards are stored
shard_dataset_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_6'

os.chdir(shard_dataset_directory)
if not os.path.isdir('./train'):
    os.mkdir('./train')
    os.mkdir('./validate')
    os.mkdir('./test')

# only get the file names that follow the shard naming convention
file_names = tf.io.gfile.glob(shard_dataset_directory + \
                              "/shard_*_of_*.tfrecords")

random.shuffle(file_names)

# first 80% of names go to the training dataset. Following 10% go to the val
# dataset, followed by last 10% go to the testing dataset.
val_split_idx = int(0.80*len(file_names))
test_split_idx = int(0.90*len(file_names))

# separate the file names out
train_files, val_files, test_files = file_names[:val_split_idx],\
                                     file_names[val_split_idx:test_split_idx],\
                                     file_names[test_split_idx:]


for sub_dir,files in zip(['/train/shard','/validate/shard','/test/shard'],
                         [train_files,val_files,test_files]):

    for file_name in files:
        new_name = file_name.split('/shard')[0] + sub_dir + file_name.split('/shard')[1]
        os.rename(file_name,new_name)


# %%
