# %% importing packages

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from skimage import measure
import cv2 as cv
import os
import tqdm
import matplotlib.pyplot as plt
import gc


# %% Citations
#############################################################
#############################################################
# https://www.tensorflow.org/guide/keras/functional
# https://www.tensorflow.org/tutorials/customization/custom_layers
# https://keras.io/examples/keras_recipes/tfrecord/
# https://arxiv.org/abs/1505.04597
# https://www.tensorflow.org/guide/gpu

# Defining Functions
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

def weighted_cce_loss(y_true,y_pred):
    '''Yes, this function essentially does what the "fit" argument 
       "class_weight" does when training a network. I had to create this 
       separate custom loss function because aparently when using tfrecord files
       for reading your dataset a check is performed comparing the input, ground
       truth, and weights values to each other. However, a comparison between 
       the empty None that is passed during the build call of the model and the
       weight array/dictionary returns an error. Thus, here is a custom loss 
       function that applies a weighting to the different classes based on the 
       distribution of the classes within the entire dataset. For thoroughness'
       sake future iteration of the dataset will only base the weights on the 
       dataset used for training, not the whole dataset.'''

    # weights for each class, as background, connective, muscle, and vasculature
    weights = [28.78661087,3.60830475,1.63037567,14.44688883]

    # create a weight for each of the images in the current batch (because the
    # weighting for categorical crossentropy needs one per input)
    for idx,weight in enumerate(weights):
        # making the input a numpy array and not an eager tensor to allow for
        # binary index masking.
        current_weights = np.asarray(tf.argmax(y_true,axis=-1)).copy().astype(
                                                                    np.float64)
        # create a mask for the current class that then becomes the value of the
        # weight. This is then passed to the loss function to apply to each
        # pixel.
        mask = current_weights==idx
        current_weights[mask] = weight

    cce = tf.keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y_true,y_pred,current_weights)

    return(cce_loss)
    
#############################################################
#############################################################

# %% Setting up the GPU, and setting memory growth to true so that it is easier
# to see how much memory the training process is taking up exactly. This code is
# from a tensorflow tutorial. 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')

    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

# use this to set mixed precision for higher efficiency later if you would like
# mixed_precision.set_global_policy('mixed_float16')


# %% setting up datasets and building model

# directory where the dataset shards are stored
shard_dataset_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_ScaleFactor2'

os.chdir(shard_dataset_directory)

# only get the file names that follow the shard naming convention
file_names = tf.io.gfile.glob(shard_dataset_directory + \
                              "/shard_*_of_*.tfrecords")

# first 70% of names go to the training dataset. Following 20% go to the val
# dataset, followed by last 10% go to the testing dataset.
val_split_idx = int(0.7*len(file_names))
test_split_idx = int(0.9*len(file_names))

# separate the file names out
train_files, val_files, test_files = file_names[:val_split_idx],\
                                     file_names[val_split_idx:test_split_idx],\
                                     file_names[test_split_idx:]

# create the datasets. Because of how batches are run for training, we set
# the dataset to repeat() because the batches and epochs are altered from 
# standard practice to fit on graphics cards and provide more meaningful and 
# frequent updates to the console.
training_dataset = get_dataset(train_files,batch_size=15)
training_dataset = training_dataset.repeat()
validation_dataset = get_dataset(val_files,batch_size = 5)
# testing has a batch size of 1 to facilitate visualization of predictions
testing_dataset = get_dataset(test_files,batch_size=1)

# %% Putting together the network

# filter multiplier provided creates largest filter depth of 256 with a 
# multiplier of 8. 
filter_multiplier = 8
# encoder convolution parameters
enc_kernel = (3,3)
enc_strides = (1,1)

# encoder max-pooling parameters
enc_pool_size = (2,2)
enc_pool_strides = (2,2)

# setting the input size
net_input = keras.Input(shape=(512,512,3),name='original_image')

################## Encoder ##################
# encoder, block 1

# including the image normalization within the network for easier image
# processing during inference
normalized = layers.Normalization()(net_input)

enc1 = layers.Conv2D(filters=2*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc1_conv1')(normalized)

enc1 = tf.keras.layers.BatchNormalization()(enc1)
enc1 = layers.ReLU()(enc1)

enc1 = layers.Conv2D(filters=2*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc1_conv2')(enc1)

enc1 = tf.keras.layers.BatchNormalization()(enc1)
enc1 = layers.ReLU()(enc1)

enc1_pool = layers.MaxPooling2D(pool_size=enc_pool_size,
                                strides=enc_pool_strides,
                                padding='same',
                                name='enc1_pool')(enc1)


# encoder, block 2
enc2 = layers.Conv2D(filters=4*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc2_conv1')(enc1_pool)

enc2 = tf.keras.layers.BatchNormalization()(enc2)
enc2 = layers.ReLU()(enc2)

enc2 = layers.Conv2D(filters=4*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc2_conv2')(enc2)

enc2 = tf.keras.layers.BatchNormalization()(enc2)
enc2 = layers.ReLU()(enc2)

enc2_pool = layers.MaxPooling2D(pool_size=enc_pool_size,
                                strides=enc_pool_strides,
                                padding='same',
                                name='enc2_pool')(enc2)


# encoder, block 3
enc3 = layers.Conv2D(filters=8*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc3_conv1')(enc2_pool)

enc3 = tf.keras.layers.BatchNormalization()(enc3)
enc3 = layers.ReLU()(enc3)
                     
enc3 = layers.Conv2D(filters=8*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc3_conv2')(enc3)

enc3 = tf.keras.layers.BatchNormalization()(enc3)
enc3 = layers.ReLU()(enc3)

enc3_pool = layers.MaxPooling2D(pool_size=enc_pool_size,
                                strides=enc_pool_strides,
                                padding='same',
                                name='enc3_pool')(enc3)                         

# encoder, block 4
enc4 = layers.Conv2D(filters=16*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc4_conv1')(enc3_pool)

enc4 = tf.keras.layers.BatchNormalization()(enc4)
enc4 = layers.ReLU()(enc4)

enc4 = layers.Conv2D(filters=16*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc4_conv2')(enc4)

enc4 = tf.keras.layers.BatchNormalization()(enc4)
enc4 = layers.ReLU()(enc4)

enc4_pool = layers.MaxPooling2D(pool_size=enc_pool_size,
                                strides=enc_pool_strides,
                                padding='same',
                                name='enc4_pool')(enc4)     


# encoder, block 5
enc5 = layers.Conv2D(filters=32*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc5_conv1')(enc4_pool)

enc5 = tf.keras.layers.BatchNormalization()(enc5)
enc5 = layers.ReLU()(enc5)

enc5 = layers.Conv2D(filters=32*filter_multiplier,
                     kernel_size=enc_kernel,
                     strides=enc_strides,
                     padding='same',
                     name='enc5_conv2')(enc5)

enc5 = tf.keras.layers.BatchNormalization()(enc5)
enc5 = layers.ReLU()(enc5)

################## Decoder ##################

# decoder upconv parameters
dec_upconv_kernel = (2,2)
dec_upconv_stride = (2,2)

# decoder forward convolution parameters
dec_conv_stride = (1,1)
dec_conv_kernel = (3,3)

# Decoder, block 4
dec4_up = layers.Conv2DTranspose(filters=16*filter_multiplier,
                              kernel_size=dec_upconv_kernel,
                              strides=dec_upconv_stride,
                              padding='same',
                              name='dec4_upconv')(enc5)

dec4_conc = layers.concatenate([dec4_up,enc4],axis=-1)

dec4 = layers.Conv2D(filters=16*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec4_conv1')(dec4_conc)

dec4 = tf.keras.layers.BatchNormalization()(dec4)
dec4 = layers.ReLU()(dec4)

dec4 = layers.Conv2D(filters=16*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec4_conv2')(dec4)

dec4 = tf.keras.layers.BatchNormalization()(dec4)
dec4 = layers.ReLU()(dec4)


# Decoder, block 3
dec3_up = layers.Conv2DTranspose(filters=8*filter_multiplier,
                              kernel_size=dec_upconv_kernel,
                              strides=dec_upconv_stride,
                              padding='same',
                              name='dec3_upconv')(dec4)

dec3_conc = layers.concatenate([dec3_up,enc3],axis=-1)

dec3 = layers.Conv2D(filters=8*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec3_conv1')(dec3_conc)

dec3 = tf.keras.layers.BatchNormalization()(dec3)
dec3 = layers.ReLU()(dec3)

dec3 = layers.Conv2D(filters=8*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec3_conv2')(dec3)

dec3 = tf.keras.layers.BatchNormalization()(dec3)
dec3 = layers.ReLU()(dec3)


# Decoder, block 2
dec2_up = layers.Conv2DTranspose(filters=4*filter_multiplier,
                              kernel_size=dec_upconv_kernel,
                              strides=dec_upconv_stride,
                              padding='same',
                              name='dec2_upconv')(dec3)

dec2_conc = layers.concatenate([dec2_up,enc2],axis=-1)

dec2 = layers.Conv2D(filters=4*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec2_conv1')(dec2_conc)

dec2 = tf.keras.layers.BatchNormalization()(dec2)
dec2 = layers.ReLU()(dec2)

dec2 = layers.Conv2D(filters=4*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec2_conv2')(dec2)

dec2 = tf.keras.layers.BatchNormalization()(dec2)
dec2 = layers.ReLU()(dec2)


# Decoder, block 1
dec1_up = layers.Conv2DTranspose(filters=2*filter_multiplier,
                              kernel_size=dec_upconv_kernel,
                              strides=dec_upconv_stride,
                              padding='same',
                              name='dec1_upconv')(dec2)

dec1_conc = layers.concatenate([dec1_up,enc1],axis=-1)

dec1 = layers.Conv2D(filters=2*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec1_conv1')(dec1_conc)

dec1 = tf.keras.layers.BatchNormalization()(dec1)
dec1 = layers.ReLU()(dec1)

dec1 = layers.Conv2D(filters=2*filter_multiplier,
                     kernel_size=dec_conv_kernel,
                     strides=dec_conv_stride,
                     padding='same',
                     name='dec1_conv2')(dec1)

dec1 = tf.keras.layers.BatchNormalization()(dec1)
dec1 = layers.ReLU()(dec1)

conv_seg = layers.Conv2D(filters=4,
                         kernel_size=(1,1),
                         name='conv_feature_map')(dec1)

prob_dist = layers.Softmax(dtype='float32')(conv_seg)

unet = keras.Model(inputs=net_input,outputs=prob_dist,name='uNet')

unet.summary()

# %% setting up training

cce = tf.keras.losses.CategoricalCrossentropy()

# running network eagerly because it allows us to use convert a tensor to a
# numpy array to help with the weighted loss calculation.
unet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=weighted_cce_loss,
    run_eagerly=True,
    metrics=[tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]                
)

# %%
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_recall',
                                                 mode='max',
                                                 factor=0.8,
                                                 patience=3,
                                                 min_lr=0.00001,
                                                 verbose=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('unet_seg_subclassed.h5',
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   monitor='val_recall',
                                                   mode='max',
                                                   verbose=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=8,
                                                     monitor='val_recall',
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=True)

num_steps = 150

history = unet.fit(training_dataset,
                   epochs=20,
                   steps_per_epoch=num_steps,
                   validation_data=validation_dataset,
                   callbacks=[checkpoint_cb,
                              early_stopping_cb,
                              reduce_lr])

# %%
# evaluate the network after loading the weights
unet.load_weights('./unet_seg_functional.h5')
results = unet.evaluate(testing_dataset)

# %%
# extracting loss vs epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
# extracting precision vs epoch
precision = history.history['precision']
val_precision = history.history['val_precision']
# extracting recall vs epoch
recall = history.history['recall']
val_recall = history.history['val_recall']

epochs = range(len(loss))

figs, axes = plt.subplots(3,1)

# plotting loss and validation loss
axes[0].plot(epochs,loss)
axes[0].plot(epochs,val_loss)
axes[0].legend(['loss','val_loss'])
axes[0].set(xlabel='epochs',ylabel='crossentropy loss')

# plotting precision and validation precision
axes[1].plot(epochs,precision)
axes[1].plot(epochs,val_precision)
axes[1].legend(['precision','val_precision'])
axes[1].set(xlabel='epochs',ylabel='precision')

# plotting recall validation recall
axes[2].plot(epochs,recall)
axes[2].plot(epochs,val_recall)
axes[2].legend(['recall','val_recall'])
axes[2].set(xlabel='epochs',ylabel='recall')


# %% exploring the predictions to better understand what the network is doing

images = []
gt = []
predictions = []

# taking out 10 of the next samples from the testing dataset and iterating 
# through them
for sample in testing_dataset.take(10):
    # make sure it is producing the correct dimensions
    print(sample[0].shape)
    # take the image and convert it back to RGB, store in list
    image = sample[0]
    image = cv.cvtColor(np.squeeze(np.asarray(image).copy()),cv.COLOR_BGR2RGB)
    images.append(image)
    # extract the ground truth and store in list
    ground_truth = sample[1]
    gt.append(ground_truth)
    # perform inference
    out = unet.predict(sample[0])
    predictions.append(out)
    # show the original input image
    plt.imshow(image)
    plt.show()
    # flatten the ground truth from one-hot encoded along the last axis, and 
    # show the resulting image
    squeezed_gt = tf.argmax(ground_truth,axis=-1)
    squeezed_prediction = tf.argmax(out,axis=-1)
    plt.imshow(squeezed_gt[0,:,:])
    # print the number of classes in this tile
    print(np.unique(squeezed_gt))
    plt.show()
    # show the flattened predictions
    plt.imshow(squeezed_prediction[0,:,:])
    print(np.unique(squeezed_prediction))
    plt.show()

# %%
# select one of the images cycled through above to investigate furtehr
image_to_investigate = 2

# show the original image
plt.imshow(images[image_to_investigate])
plt.show()

# show the ground truth for this tile
squeezed_gt = tf.argmax(gt[image_to_investigate],axis=-1)
plt.imshow(squeezed_gt[0,:,:])
# print the number of unique classes in the ground truth
print(np.unique(squeezed_gt))
plt.show()
 # flatten the prediction and show the probability distribution
squeezed_prediction = tf.argmax(predictions[image_to_investigate],axis=-1)
plt.imshow(predictions[image_to_investigate][0,:,:,3])
plt.show()
# show the flattened image
plt.imshow(squeezed_prediction[0,:,:])
print(np.unique(squeezed_prediction))
plt.show()
