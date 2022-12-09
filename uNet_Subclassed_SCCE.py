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
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]
# you can alternatively call this script using this line in the terminal to
# address the issue of memory leak when using the dataset.shuffle buffer. Found
# at the subsequent link.
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9 python3 uNet_Subclassed.py

# https://stackoverflow.com/questions/55211315/memory-leak-with-tf-data/66971031#66971031


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

    # note that the bounding boxes are included here, but are not used. These 
    # were included in the dataset for future use if I wanted to put together
    # something like YOLO for practice. Could be used later, but also haven't 
    # been thoroughly tested, so could be buggy and should be vetted.
    bbox_x = content['bbox_x']
    bbox_y = content['bbox_y']
    bbox_height = content['bbox_height']
    bbox_width = content['bbox_width']

    # convert the images to uint8, and reshape them accordingly
    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    image = tf.reshape(image,shape=[height,width,3])
    segmentation = tf.io.parse_tensor(raw_seg, out_type=tf.uint8)-1
    # This is including the class weights in the parser, enabling them to be
    # used by the loss function to weight the loss and accuracy metrics.
    # Note that the last two are divided by two to prevent them from being over
    # segmented, which they were.
    # [2.72403952, 2.81034368, 4.36437716, 36.66264202, 108.40694198, 87.39903838]
    weights = [2.72403952*2,
               2.81034368, 
               4.36437716, 
               36.66264202, 
               108.40694198/2, 
               87.39903838/2]
    weights = np.divide(weights,sum(weights))
    
    # the weights are calculated by the tf_record_weight_determination.py file,
    # and are related to the percentages of each class in the dataset.
    sample_weights = tf.gather(weights, indices=tf.cast(segmentation, tf.int32))

    return(image,segmentation,sample_weights)

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
        self.image_normalization = layers.Rescaling(scale=1./255)

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

    def call(self,input,normalization=False,training=True,include_pool=True):
        
        # first conv of the encoder block
        if normalization:
            x = self.image_normalization(input)
            x = self.conv1(x)
        else:
            x = self.conv1(input)

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


    def call(self,input,skip_conn,training=True,segmentation=False,prob_dist=True):
        
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
            if prob_dist:
                seg = layers.Softmax(dtype='float32')(seg)

            return(seg)

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


    def call(self,inputs,training,predict=False,threshold=3):

        # encoder    
        enc1,enc1_pool = self.encoder_block1(input=inputs,normalization=True,training=training)
        enc2,enc2_pool = self.encoder_block2(input=enc1_pool,training=training)
        enc3,enc3_pool = self.encoder_block3(input=enc2_pool,training=training)
        enc4,enc4_pool = self.encoder_block4(input=enc3_pool,training=training)
        enc5 = self.encoder_block5(input=enc4_pool,
                                   include_pool=False,
                                   training=training)

        # enc4 = self.encoder_block4(input=enc3_pool,
        #                            include_pool=False,
        #                            training=training)


        # decoder
        dec4 = self.decoder_block4(input=enc5,skip_conn=enc4,training=training)
        dec3 = self.decoder_block3(input=dec4,skip_conn=enc3,training=training)
        dec2 = self.decoder_block2(input=dec3,skip_conn=enc2,training=training)
        prob_dist_out = self.decoder_block1(input=dec2,
                                            skip_conn=enc1,
                                            segmentation=True,
                                            training=training)
        if predict:
            seg_logits_out = self.decoder_block1(input=dec2,
                                                 skip_conn=enc1,
                                                 segmentation=True,
                                                 training=training,
                                                 prob_dist=False)

        # This prediction is included to allow one to seta threshold for the 
        # uncertainty, deemed an arbitrary value that corresponds to the 
        # maximum value of the logits predicted at a specific point in the 
        # image. It only includes predictions for the vascular and neural 
        # tissues if they are above the confidence threshold, if they are below
        # the threshold the predictions are defaulted to muscle, connective,
        # or background.
        
        if predict:
            # rename the value for consistency and write protection.
            y_pred = seg_logits_out
            pred_shape = (1,1024,1024,6)
            # Getting an image-sized preliminary segmentation prediction
            squeezed_prediction = tf.squeeze(tf.argmax(y_pred,axis=-1))

            # initializing the variable used for storing the maximum logits at 
            # each pixel location.
            max_value_predictions = tf.zeros((1024,1024))

            # cycle through all the classes 
            for idx in range(6):
                
                # current class logits
                current_slice = tf.squeeze(y_pred[:,:,:,idx])
                # find the locations where this class is predicted
                current_indices = squeezed_prediction == idx
                # define the shape so that this function can run in graph mode
                # and not need eager execution.
                current_indices.set_shape((1024,1024))
                # Get the indices of where the idx class is predicted
                indices = tf.where(squeezed_prediction == idx)
                # get the output of boolean_mask to enable scatter update of the
                # tensor. This is required because tensors do not support 
                # mask indexing.
                values_updates = tf.boolean_mask(current_slice,current_indices).astype(tf.double)
                # Place the maximum logit values at each point in an 
                # image-size matrix, indicating the confidence in the prediction
                # at each pixel. 
                max_value_predictions = tf.tensor_scatter_nd_update(max_value_predictions,indices,values_updates.astype(tf.float32))
            
            for idx in [3,4]:
                mask_list = []
                for idx2 in range(6):
                    if idx2 == idx:
                        mid_mask = max_value_predictions<threshold
                        mask_list.append(mid_mask.astype(tf.float32))
                    else:
                        mask_list.append(tf.zeros((1024,1024)))

                mask = tf.expand_dims(tf.stack(mask_list,axis=-1),axis=0)

                indexes = tf.where(mask)
                values_updates = tf.boolean_mask(tf.zeros(pred_shape),mask).astype(tf.double)

                seg_logits_out = tf.tensor_scatter_nd_update(seg_logits_out,indexes,values_updates.astype(tf.float32))
                prob_dist_out = layers.Softmax(dtype='float32')(seg_logits_out)
            # print("updated logits!")


            
        return(prob_dist_out)


    # def test_step(self, data):
        
    #     threshold = 3
    #     x, y, weight = data
    #     pred_shape = (1,1024,1024,6)

    #     y_pred = self(x,training=False)

    #     squeezed_prediction = tf.squeeze(tf.argmax(y_pred,axis=-1))

    #     max_value_predictions = tf.zeros((1024,1024))

    #     for idx in range(6):

    #         current_slice = tf.squeeze(y_pred[:,:,:,idx])
    #         current_indices = squeezed_prediction == idx
    #         current_indices.set_shape((1024,1024))
    #         indices = tf.where(squeezed_prediction == idx)
    #         values_updates = tf.boolean_mask(current_slice,current_indices).astype(tf.double)
    #         max_value_predictions = tf.tensor_scatter_nd_update(max_value_predictions,indices,values_updates.astype(tf.float32))
        
    #     for idx in [3,4]:
    #         mask_list = []
    #         for idx2 in range(6):
    #             if idx2 == idx:
    #                 mid_mask = max_value_predictions<threshold
    #                 mask_list.append(mid_mask.astype(tf.float32))
    #             else:
    #                 mask_list.append(tf.zeros((1024,1024)))

    #         mask = tf.expand_dims(tf.stack(mask_list,axis=-1),axis=0)

    #         indexes = tf.where(mask)
    #         values_updates = tf.boolean_mask(tf.zeros(pred_shape),mask).astype(tf.double)

    #         y_pred = tf.tensor_scatter_nd_update(y_pred,indexes,values_updates.astype(tf.float32))

    #     self.compiled_metrics.update_state(y, y_pred, sample_weight=weight)
    #     self.compiled_loss(y, y_pred, sample_weight=weight)

    #     return {m.name: m.result() for m in self.metrics}

#############################################################

class SanityCheck(keras.callbacks.Callback):

    def __init__(self, testing_images):
        super(SanityCheck, self).__init__()
        self.testing_images = testing_images


    def on_epoch_end(self,epoch, logs=None):
        for image_pair in self.testing_images:
            out = self.model.predict(image_pair[0],verbose=0)
            image = cv.cvtColor(np.squeeze(np.asarray(image_pair[0]).copy()),cv.COLOR_BGR2RGB)
            squeezed_gt = image_pair[1][0,:,:]
            squeezed_prediction = tf.argmax(out,axis=-1)

            fig,ax = plt.subplots(1,3)

            ax[0].imshow(image)
            ax[1].imshow(squeezed_gt,vmin=0, vmax=5)
            ax[2].imshow(squeezed_prediction[0,:,:],vmin=0, vmax=5)

            plt.show()
            print(np.unique(squeezed_gt))
            print(np.unique(squeezed_prediction[0,:,:]))


#############################################################

def load_dataset(file_names):
    '''Receives a list of file names from a folder that contains tfrecord files
       compiled previously. Takes these names and creates a tensorflow dataset
       from them.'''

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(file_names)

    # you can shard the dataset if you like to reduce the size when necessary
    dataset = dataset.shard(num_shards=8,index=2)
    
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
    dataset = dataset.shuffle(300)

    # adding the batch size to the dataset
    dataset = dataset.batch(batch_size=batch_size)

    return(dataset)


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
home_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_6'
training_directory = home_directory + '/train'
val_directory = home_directory + '/validate'
testing_directory = home_directory + '/test'

os.chdir(home_directory)

# only get the file names that follow the shard naming convention
train_files = tf.io.gfile.glob(training_directory + \
                              "/shard_*_of_*.tfrecords")
val_files = tf.io.gfile.glob(val_directory + \
                              "/shard_*_of_*.tfrecords")
test_files = tf.io.gfile.glob(testing_directory + \
                              "/shard_*_of_*.tfrecords")

# create the datasets. Because of how batches are run for training, we set
# the dataset to repeat() because the batches and epochs are altered from 
# standard practice to fit on graphics cards and provide more meaningful and 
# frequent updates to the console.
training_dataset = get_dataset(train_files,batch_size=3)
training_dataset = training_dataset.repeat()
validation_dataset = get_dataset(val_files,batch_size = 3)
# testing has a batch size of 1 to facilitate visualization of predictions
testing_dataset = get_dataset(test_files,batch_size=1)

# explicitly puts the model on the GPU to show how large it is. 
gpus = tf.config.list_logical_devices('GPU')
with tf.device(gpus[0].name):
    # filter multiplier provided creates largest filter depth of 256 with a 
    # multiplier of 8. 
    sample_data = np.zeros((1,1024,1024,3)).astype(np.int8)
    unet = uNet(filter_multiplier=12,) # 12 is the magic number
    # build with input image size of 512*512
    out = unet(sample_data)
    unet.summary()
# %%

unet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    run_eagerly=False,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

test_images = []
for sample in testing_dataset.take(5):
    #print(sample[0].shape)
    test_images.append([sample[0],sample[1]])

sanity_check = SanityCheck(test_images)


def schedule(epoch, lr): 
        return(lr*0.97)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 mode='min',
                                                 factor=0.8,
                                                 patience=5,
                                                 min_lr=0.000001,
                                                 verbose=True,
                                                 min_delta=0.01,)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'unet_seg_weights.{epoch:02d}-{val_sparse_categorical_accuracy:.4f}-{val_loss:.4f}.h5',
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    verbose=True
    )

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20,
                                                     monitor='val_sparse_categorical_accuracy',
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=True,
                                                     min_delta=0.001)

# setting the number of batches to iterate through each epoch to a value much
# lower than what it normaly would be so that we can actually see what is going
# on with the network, as well as have a meaningful early stopping.


# %% fit the network!
num_steps = 600

history = unet.fit(training_dataset,
                   epochs=100,
                   steps_per_epoch=num_steps,
                   validation_data=validation_dataset,
                   verbose=2,
                   callbacks=[checkpoint_cb,
                              early_stopping_cb,
                              lr_scheduler,])
# %%



# %%
# evaluate the network after loading the weights
unet.load_weights('unet_seg_weights.77-0.9175-0.0082.h5')
results = unet.evaluate(testing_dataset)
print(results)
# %%
# extracting loss vs epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

# extracting precision vs epoch

epochs = range(len(loss))

figs, axes = plt.subplots(2,1)

# plotting loss and validation loss
axes[0].plot(epochs[1:],loss[1:])
axes[0].plot(epochs[1:],val_loss[1:])
axes[0].legend(['loss','val_loss'])
axes[0].set(xlabel='epochs',ylabel='crossentropy loss')

# plotting loss and validation loss
axes[1].plot(epochs[1:],acc[1:])
axes[1].plot(epochs[1:],val_acc[1:])
axes[1].legend(['acc','val_acc'])
axes[1].set(xlabel='epochs',ylabel='weighted accuracy')


# %% exploring the predictions to better understand what the network is doing. 
# This section is largely experimental, and should be treated as such. I have
# included it in this network file for the sake of documentation and 
# traceability, but it is not in the other network files for full image 
# segmentation and directory segmentation because, well, those are functional 
# and this is experimental.


# uncomment everything from here down to use this section
images = []
gt = []
predictions = []
# higher threshold means the network must be more confident.
threshold = 3

# taking out 15 of the next samples from the testing dataset and iterating 
# through them
for sample in testing_dataset.take(15):
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
    out = unet(sample[0],predict=True,threshold=threshold)
    predictions.append(out)
    # show the original input image
    plt.imshow(image)
    plt.show()
    # flatten the ground truth from one-hot encoded along the last axis, and 
    # show the resulting image
    squeezed_gt = ground_truth
    squeezed_prediction = tf.argmax(out,axis=-1)
    plt.imshow(squeezed_gt[0,:,:],vmin=0, vmax=5)
    # print the number of classes in this tile
    print(np.unique(squeezed_gt))
    plt.show()
    # show the flattened predictions
    plt.imshow(squeezed_prediction[0,:,:],vmin=0, vmax=5)
    print(np.unique(squeezed_prediction))
    plt.show()

# # %% 5, 6, 8
# # select one of the images cycled through above to investigate further
# image_to_investigate = 0
# threshold = 2
# # show the original image
# plt.imshow(images[image_to_investigate])
# plt.show()

# # show the ground truth for this tile
# squeezed_gt = gt[image_to_investigate]
# plt.imshow(squeezed_gt[0,:,:])
# # print the number of unique classes in the ground truth
# print(np.unique(squeezed_gt))
# plt.show()
#  # flatten the prediction and show the probability distribution

# out = predictions[image_to_investigate]


# # plt.hist(out[:,:,:,4].reshape(-1),alpha=0.5,label='neural')
# # plt.hist(out[:,:,:,3].reshape(-1),alpha=0.5,label='vascular')
# # plt.legend(["neural",'vascular'])

# out = predictions[image_to_investigate]
# squeezed_prediction = np.squeeze(tf.argmax(out,axis=-1))

# max_value_predictions = np.zeros(squeezed_prediction.shape)

# for idx in range(6):
#     current_slice = np.squeeze(out[:,:,:,idx])
#     current_indices = squeezed_prediction == idx
#     indices = tf.where(squeezed_prediction == idx)
#     values_updates = tf.boolean_mask(current_slice,current_indices).astype(tf.double)
#     max_value_predictions = tf.tensor_scatter_nd_update(max_value_predictions,indices,values_updates.astype(tf.float32))

# plt.imshow(max_value_predictions)
# plt.show()

# for idx in [3,4]:
#     mask = np.zeros(out.shape)
#     mask[:,:,:,idx] = max_value_predictions<threshold
#     indices = tf.where(mask)
#     values_updates = tf.boolean_mask(np.zeros(out.shape),mask).astype(tf.double)

#     out = tf.tensor_scatter_nd_update(out,indices,values_updates.astype(tf.float32))

# for idx in range(6):
#     current_slice = np.squeeze(out[:,:,:,idx])
#     current_indices = squeezed_prediction == idx
#     indices = tf.where(squeezed_prediction == idx)
#     values_updates = tf.boolean_mask(current_slice,current_indices).astype(tf.double)
#     max_value_predictions = tf.tensor_scatter_nd_update(max_value_predictions,indices,values_updates.astype(tf.float32))
# plt.imshow(max_value_predictions)
# plt.show()


# squeezed_prediction = tf.argmax(predictions[image_to_investigate],axis=-1)
# # plt.imshow(predictions[image_to_investigate][0,:,:,3])
# # plt.show()
# # show the flattened image
# plt.imshow(squeezed_prediction[0,:,:])
# print(np.unique(squeezed_prediction))
# plt.show()

# squeezed_prediction = tf.argmax(out,axis=-1)
# # plt.imshow(predictions[image_to_investigate][0,:,:,3])
# # plt.show()
# # show the flattened image
# plt.imshow(squeezed_prediction[0,:,:])
# print(np.unique(squeezed_prediction))
# plt.show()

# # %%
# image_to_investigate = 0
# threshold = 1
# y_pred = predictions[image_to_investigate]


# pred_shape = (1,1024,1024,6)

# squeezed_prediction = tf.squeeze(tf.argmax(y_pred,axis=-1))

# max_value_predictions = tf.zeros((1024,1024))

# for idx in range(6):

#     current_slice = tf.squeeze(y_pred[:,:,:,idx])
#     current_indices = squeezed_prediction == idx
#     current_indices.set_shape((1024,1024))
#     indices = tf.where(squeezed_prediction == idx)
#     values_updates = tf.boolean_mask(current_slice,current_indices).astype(tf.double)
#     max_value_predictions = tf.tensor_scatter_nd_update(max_value_predictions,indices,values_updates.astype(tf.float32))

# for idx in [3,4]:
#     mask_list = []
#     for idx2 in range(6):
#         if idx2 == idx:
#             mid_mask = max_value_predictions<threshold
#             mask_list.append(mid_mask.astype(tf.float32))
#         else:
#             mask_list.append(tf.zeros((1024,1024)))

#     mask = tf.expand_dims(tf.stack(mask_list,axis=-1),axis=0)

#     indexes = tf.where(mask)
#     values_updates = tf.boolean_mask(tf.zeros(pred_shape),mask).astype(tf.double)

#     y_pred = tf.tensor_scatter_nd_update(y_pred,indexes,values_updates.astype(tf.float32))

# squeezed_prediction = tf.argmax(predictions[image_to_investigate],axis=-1)
# # plt.imshow(predictions[image_to_investigate][0,:,:,3])
# # plt.show()
# # show the flattened image
# plt.imshow(squeezed_prediction[0,:,:])
# print(np.unique(squeezed_prediction))
# plt.show()

# squeezed_prediction = tf.argmax(y_pred,axis=-1)
# # plt.imshow(predictions[image_to_investigate][0,:,:,3])
# # plt.show()
# # show the flattened image
# plt.imshow(squeezed_prediction[0,:,:])
# print(np.unique(squeezed_prediction))
# plt.show()
# # %%

# %%
