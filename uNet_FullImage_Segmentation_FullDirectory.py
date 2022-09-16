# %% importing packages

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from skimage import measure
from skimage import morphology
from scipy import ndimage
import cv2 as cv
import os
import matplotlib.pyplot as plt
import tqdm
from natsort import natsorted
plt.rcParams['figure.figsize'] = [50, 150]


# %% Citations
#############################################################
#############################################################


# Defining Functions
#############################################################
#############################################################

class EncoderBlock(layers.Layer):
    '''This function returns an encoder block with two convolutional layers and 
       an option for returning both a max-pooled output with a stride and pool 
       size of (2,2) and the output of the second convolution for skip 
       connections implemented later in the network during the decoding 
       section. All padding is set to "same" for cleanliness.
       
       When initializing it receives the number of filters to be used in both
       of the convolutional layers as well as the kernel size and stride for 
       those same layers. It also receives the trainable variable for use with
       the batch normalization layers.'''

    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 strides=(1,1),
                 trainable=True,
                 name='encoder_block',
                 **kwargs):

        super(EncoderBlock,self).__init__(trainable, name, **kwargs)
        # When initializing this object receives a trainable parameter for
        # freezing the convolutional layers. 

        # including the image normalization within the network for easier image
        # processing during inference
        self.image_normalization = layers.Normalization()

        # below creates the first of two convolutional layers
        self.conv1 = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      name='encoder_conv1',
                      trainable=trainable)

        # second of two convolutional layers
        self.conv2 = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      name='encoder_conv2',
                      trainable=trainable)

        # creates the max-pooling layer for downsampling the image.
        self.enc_pool = layers.MaxPool2D(pool_size=(2,2),
                                    strides=(2,2),
                                    padding='same',
                                    name='enc_pool')

        # ReLU layer for activations.
        self.ReLU = layers.ReLU()
        
        # both batch normalization layers for use with their corresponding
        # convolutional layers.
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

    def call(self,input,training=True,include_pool=True):
        
        # first conv of the encoder block
        x = self.image_normalization(input)
        x = self.conv1(x)
        x = self.batch_norm1(x,training=training)
        x = self.ReLU(x)

        # second conv of the encoder block
        x = self.conv2(x)
        x = self.batch_norm2(x,training=training)
        x = self.ReLU(x)
        
        # calculate and include the max pooling layer if include_pool is true.
        # This output is used for the skip connections later in the network.
        if include_pool:
            pooled_x = self.enc_pool(x)
            return(x,pooled_x)

        else:
            return(x)


#############################################################

class DecoderBlock(layers.Layer):
    '''This function returns a decoder block that when called receives both an
       input and a "skip connection". The input is passed to the 
       "up convolution" or transpose conv layer to double the dimensions before
       being concatenated with its associated skip connection from the encoder
       section of the network. All padding is set to "same" for cleanliness. 
       The decoder block also has an option for including an additional 
       "segmentation" layer, which is a (1,1) convolution with 4 filters, which
       produces the logits for the one-hot encoded ground truth. 
       
       When initializing it receives the number of filters to be used in the
       up convolutional layer as well as the other two forward convolutions. 
       The received kernel_size and stride is used for the forward convolutions,
       with the up convolution kernel and stride set to be (2,2).'''
    def __init__(self,
                 filters,
                 trainable=True,
                 kernel_size=(3,3),
                 strides=(1,1),
                 name='DecoderBlock',
                 **kwargs):

        super(DecoderBlock,self).__init__(trainable, name, **kwargs)

        # creating the up convolution layer
        self.up_conv = layers.Conv2DTranspose(filters=filters,
                                              kernel_size=(2,2),
                                              strides=(2,2),
                                              padding='same',
                                              name='decoder_upconv',
                                              trainable=trainable)

        # the first of two forward convolutional layers
        self.conv1 = layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='same',
                                   name ='decoder_conv1',
                                   trainable=trainable)

        # second convolutional layer
        self.conv2 = layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='same',
                                   name ='decoder_conv2',
                                   trainable=trainable)

        # this creates the output prediction logits layer.
        self.seg_out = layers.Conv2D(filters=6,
                        kernel_size=(1,1),
                        name='conv_feature_map')

        # ReLU for activation of all above layers
        self.ReLU = layers.ReLU()
        
        # the individual batch normalization layers for their respective 
        # convolutional layers.
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()


    def call(self,input,skip_conn,training=True,segmentation=False):
        
        up = self.up_conv(input) # perform image up convolution
        # concatenate the input and the skip_conn along the features axis
        concatenated = layers.concatenate([up,skip_conn],axis=-1)

        # first convolution 
        x = self.conv1(concatenated)
        x = self.batch_norm1(x,training=training)
        x = self.ReLU(x)

        # second convolution
        x = self.conv2(x)
        x = self.batch_norm2(x,training=training)
        x = self.ReLU(x)

        # if segmentation is True, then run the segmentation (1,1) convolution
        # and use the Softmax to produce a probability distribution.
        if segmentation:
            seg = self.seg_out(x)
            # deliberately set as "float32" to ensure proper calculation if 
            # switching to mixed precision for efficiency
            prob = layers.Softmax(dtype='float32')(seg)
            return(prob)

        else:
            return(x)


#############################################################

class uNet(keras.Model):
    '''This is a sub-classed model that uses the encoder and decoder blocks
       defined above to create a custom unet. The differences from the original 
       paper include a variable filter scalar (filter_multiplier), batch 
       normalization between each convolutional layer and the associated ReLU 
       activation, as well as feature normalization implemented in the first 
       layer of the network.'''
    def __init__(self,filter_multiplier=2,**kwargs):
        super(uNet,self).__init__()
        
        # Defining encoder blocks
        self.encoder_block1 = EncoderBlock(filters=2*filter_multiplier,
                                           name='Enc1')
        self.encoder_block2 = EncoderBlock(filters=4*filter_multiplier,
                                           name='Enc2')
        self.encoder_block3 = EncoderBlock(filters=8*filter_multiplier,
                                           name='Enc3')
        self.encoder_block4 = EncoderBlock(filters=16*filter_multiplier,
                                           name='Enc4')
        self.encoder_block5 = EncoderBlock(filters=32*filter_multiplier,
                                           name='Enc5')

        # Defining decoder blocks. The names are in reverse order to make it 
        # (hopefully) easier to understand which skip connections are associated
        # with which decoder layers.
        self.decoder_block4 = DecoderBlock(filters=16*filter_multiplier,
                                           name='Dec4')
        self.decoder_block3 = DecoderBlock(filters=8*filter_multiplier,
                                           name='Dec3')
        self.decoder_block2 = DecoderBlock(filters=4*filter_multiplier,
                                           name='Dec2')
        self.decoder_block1 = DecoderBlock(filters=2*filter_multiplier,
                                           name='Dec1')


    def call(self,inputs,training):

        # encoder    
        enc1,enc1_pool = self.encoder_block1(input=inputs,training=training)
        enc2,enc2_pool = self.encoder_block2(input=enc1_pool,training=training)
        enc3,enc3_pool = self.encoder_block3(input=enc2_pool,training=training)
        enc4,enc4_pool = self.encoder_block4(input=enc3_pool,training=training)
        enc5 = self.encoder_block5(input=enc4_pool,
                                   include_pool=False,
                                   training=training)

        # decoder
        dec4 = self.decoder_block4(input=enc5,skip_conn=enc4,training=training)
        dec3 = self.decoder_block3(input=dec4,skip_conn=enc3,training=training)
        dec2 = self.decoder_block2(input=dec3,skip_conn=enc2,training=training)
        seg_logits_out = self.decoder_block1(input=dec2,
                                             skip_conn=enc1,
                                             segmentation=True,
                                             training=training)

        return(seg_logits_out)

#############################################################

def get_image_blocks(image,tile_distance=512,tile_size=1024):
    '''Receives an image as well as a minimum distance between tiles. 
       Returns the name of the image processed, the image dimensions, and a list
       of tile centers evenly distributed across the tissue surface.'''
    image_dimensions = image.shape

    safe_mask = np.zeros([image_dimensions[0],image_dimensions[1]])
    safe_mask[int(tile_size/2):image_dimensions[0]-int(tile_size/2),
              int(tile_size/2):image_dimensions[1]-int(tile_size/2)] = 1

    grid_0 = np.arange(0,image_dimensions[0],tile_distance)
    grid_1 = np.arange(0,image_dimensions[1],tile_distance)

    

    center_indexes = []

    for grid0 in grid_0:
        for grid1 in grid_1:
            if safe_mask[grid0,grid1]:
                center_indexes.append([grid0,grid1])

    return([image_dimensions,center_indexes])

#############################################################

def get_reduced_tile_indexes(tile_center,returned_size=1024):
    start_0 = int(tile_center[0] - returned_size/2)
    end_0 = int(tile_center[0] + returned_size/2)

    start_1 = int(tile_center[1] - returned_size/2)
    end_1 = int(tile_center[1] + returned_size/2)

    return([start_0,end_0],[start_1,end_1])

#############################################################

def segment_tiles(unet,center_indexes,image,scaling_factor=1,tile_size=1024):
    
    m,n,z = image.shape
    segmentation = np.zeros((m,n))

    for idx in tqdm.tqdm(range(len(center_indexes))):
        center = center_indexes[idx]
        dim0, dim1 = get_reduced_tile_indexes(center,tile_size)
        sub_sectioned_tile = image[dim0[0]:dim0[1],dim1[0]:dim1[1]] 

        full_tile_dim0,full_tile_dim1,z = sub_sectioned_tile.shape

        color_tile = sub_sectioned_tile[:,:,0:3]

        if scaling_factor > 1:
            height = color_tile.shape[0]
            width = color_tile.shape[1]

            height2 = int(height/scaling_factor)
            width2 = int(width/scaling_factor)
            
            color_tile = cv.resize(color_tile,[height2,width2],cv.INTER_AREA)

        color_tile = color_tile[None,:,:,:]

        prediction = unet.predict(color_tile,verbose=0)

        prediction_tile = np.squeeze(np.asarray(tf.argmax(prediction,axis=-1)).astype(np.float32).copy())

        if scaling_factor > 1:
            prediction_tile = cv.resize(prediction_tile,[full_tile_dim0,full_tile_dim1],cv.INTER_NEAREST)


        dim0, dim1 = get_reduced_tile_indexes(center,returned_size=512)

        # fix this hard coding of the tile indexes for the prediction
        segmentation[dim0[0]:dim0[1],dim1[0]:dim1[1]] = prediction_tile[256:768,256:768]

    return(segmentation)

#############################################################

def segment_directory(JPG_directory,
                      unet,tile_size=2048,
                      tile_distance=512,
                      scaling_factor=2
                      ):
    os.chdir(JPG_directory)

    out_directory = './../uNet_Segmentations/'

    # create the directory for saving if it doesn't already exist
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    os.chdir(out_directory)

    file_names = tf.io.gfile.glob(JPG_directory + '*.jpg')

    for idx,file in enumerate(file_names):
        print(f'segmenting file {idx} of {len(file_names)}')

        file_id = file.split('/')[-1].split('.')[0]

        image = cv.imread(file,cv.IMREAD_UNCHANGED)
        image = cv.copyMakeBorder(image,4000,4000,4000,4000,cv.BORDER_REPLICATE)

        dimensions,center_indexes = get_image_blocks(image,
                                                    tile_distance=tile_distance,
                                                    tile_size=tile_size
                                                    )

        segmentation = segment_tiles(unet,
                                    center_indexes,
                                    image,
                                    scaling_factor=scaling_factor,
                                    tile_size=tile_size)

        cv.imwrite(
            file_id + 
            f'_uNetSegmentation.png',
            segmentation
            )

    return()


#############################################################

def double_check_produced_dataset(new_directory,image_idx=0):
    '''this function samples a random image from a given directory, crops off 
       the ground truth from the 4th layer, and displays the color image to 
       verify they work.'''
    os.chdir(new_directory)
    file_names = tf.io.gfile.glob('./*.png')
    file_names = natsorted(file_names)
    # pick a random image index number
    if image_idx == 0:
        image_idx = int(np.random.random()*len(file_names))
    else:
        pass

    print(image_idx)
    # reading specific file from the random index
    segmentation = cv.imread(file_names[image_idx],cv.IMREAD_UNCHANGED)
    # changing the color for the tile from BGR to RGB
    print(file_names[image_idx])
    # plotting the images next to each other
    plt.imshow(segmentation,vmin=0, vmax=6)
    print(np.unique(segmentation))
    plt.show()


#############################################################
#############################################################
# %%
tile_size = 4096
unet_directory =  '/home/briancottle/Research/Semantic_Segmentation/dataset_shards/'
os.chdir(unet_directory)

sample_data = np.zeros((1,1024,1024,3)).astype(np.int8)
unet = uNet(filter_multiplier=16)
out = unet(sample_data)
unet.summary()

unet.load_weights('../uNet_Networks (Best 50-0.41-0.95)/unet_seg_weights.50-0.41-0.95.h5')

# %%

JPG_directory = '/var/confocaldata/HumanNodal/HeartData/08/02/JPG/'

segment_directory(JPG_directory,
                  unet,
                  tile_size=tile_size,
                  tile_distance=512,
                  scaling_factor=4
                )


# %%

double_check_produced_dataset('/var/confocaldata/HumanNodal/HeartData/08/02/uNet_Segmentations',
                              image_idx=0)

# %%
