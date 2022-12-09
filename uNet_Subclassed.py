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
        self.seg_out = layers.Conv2D(filters=7,
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
        # self.encoder_block5 = EncoderBlock(filters=32*filter_multiplier,
        #                                    name='Enc5')

        # Defining decoder blocks. The names are in reverse order to make it 
        # (hopefully) easier to understand which skip connections are associated
        # with which decoder layers.
        # self.decoder_block4 = DecoderBlock(filters=16*filter_multiplier,
        #                                    name='Dec4')
        self.decoder_block3 = DecoderBlock(filters=8*filter_multiplier,
                                           name='Dec3')
        self.decoder_block2 = DecoderBlock(filters=4*filter_multiplier,
                                           name='Dec2')
        self.decoder_block1 = DecoderBlock(filters=2*filter_multiplier,
                                           name='Dec1')


    def call(self,inputs,training):

        # encoder    
        enc1,enc1_pool = self.encoder_block1(input=inputs,normalization=True,training=training)
        enc2,enc2_pool = self.encoder_block2(input=enc1_pool,training=training)
        enc3,enc3_pool = self.encoder_block3(input=enc2_pool,training=training)

        # enc4,enc4_pool = self.encoder_block4(input=enc3_pool,training=training)
        # enc5 = self.encoder_block5(input=enc4_pool,
        #                            include_pool=False,
        #                            training=training)

        enc4 = self.encoder_block4(input=enc3_pool,
                                   include_pool=False,
                                   training=training)


        # decoder
        # dec4 = self.decoder_block4(input=enc5,skip_conn=enc4,training=training)
        dec3 = self.decoder_block3(input=enc4,skip_conn=enc3,training=training)
        dec2 = self.decoder_block2(input=dec3,skip_conn=enc2,training=training)
        seg_logits_out = self.decoder_block1(input=dec2,
                                             skip_conn=enc1,
                                             segmentation=True,
                                             training=training)

        return(seg_logits_out)

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

def weighted_cce_loss(y_true,y_pred):
    '''Yes, this function essentially does what the "fit" argument 
       "class_weight" does when training a network. I had to create this 
       separate custom loss function because aparently when using tfrecord files
       for reading your dataset a check is performed comparing the input, ground
       truth, and weights values to each other. However, a comparison between 
       the empty None that is passed during the build call of the model and the
       weight array/dictionary returns an error. Thus, here is a custom loss 
       function that applies a weighting to the different classes based on the 
       distribution of the classes within the entire dataset. Note that the 
       weights used here are only from the training set, not including images
       from the testing and validation sets, to prevent any over-eager reviewers
       from screaming "information leak!!"
       Just kidding, it is first to prevent an information leak, and second to 
       preempt over-eager reviewers.'''


       

    # weights for each class, as background, connective, muscle, and vasculature
    # weights = [0, 2.95559004,   7.33779693,  12.87393959, 1000.43461107, 1200.63780628, 20.23600735]
    # weights = [0, 0.80284233, 1.68275694, 2.63726432, 3000.8055788, 2000.26933614, 100.30741485] # last good run
    # [0,2.72403952, 2.81034368, 4.36437716, 36.66264202, 108.40694198, 87.39903838]
    weights = [0,2.72403952, 2.81034368, 4.36437716, 36.66264202, 108.40694198, 87.39903838]

    count = 0


    all_weights_for_loss = tf.expand_dims(tf.ones((1024,1024)).astype(tf.float64), axis=0)

    for image in y_true:
        weights_for_image = tf.ones((1024,1024)).astype(tf.float64)

        for idx,weight in enumerate(weights):
            mask = image[:,:,idx]
            mask.set_shape((1024,1024))
            indexes = tf.where(mask)
            values_mask = mask*weights[idx]

            values_updates = tf.boolean_mask(values_mask,mask).astype(tf.double)

            weights_for_image = tf.tensor_scatter_nd_update(weights_for_image,indexes,values_updates)

        if count == 0:
            all_weights_for_loss = tf.expand_dims(weights_for_image, axis=0)
        else:
            all_weights_for_loss = tf.concat([all_weights_for_loss,tf.expand_dims(weights_for_image, axis=0)],axis=0)
        count += 1

    cce = tf.keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y_true,y_pred,all_weights_for_loss)

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
os.chdir('/home/briancottle/Research/Semantic_Segmentation/dataset_shards_5/')
training_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_5/train'
val_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_5/validate'
testing_directory = '/home/briancottle/Research/Semantic_Segmentation/dataset_shards_5/test'

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
training_dataset = get_dataset(train_files,batch_size=1)
training_dataset = training_dataset.repeat()
validation_dataset = get_dataset(val_files,batch_size = 1)
# testing has a batch size of 1 to facilitate visualization of predictions
testing_dataset = get_dataset(test_files,batch_size=1)

# explicitly puts the model on the GPU to show how large it is. 
gpus = tf.config.list_logical_devices('GPU')
with tf.device(gpus[0].name):
    # filter multiplier provided creates largest filter depth of 256 with a 
    # multiplier of 8. 
    sample_data = np.zeros((1,1024,1024,3)).astype(np.int8)
    unet = uNet(filter_multiplier=32,)
    # build with input image size of 512*512
    out = unet(sample_data)
    unet.summary()
# %%
# running network eagerly because it allows us to use convert a tensor to a
# numpy array to help with the weighted loss calculation.
unet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    run_eagerly=True,
    metrics=[tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
)

# %%
class SanityCheck(keras.callbacks.Callback):

    def __init__(self, testing_images):
        super(SanityCheck, self).__init__()
        self.testing_images = testing_images


    def on_epoch_end(self,epoch, logs=None):
        for image_pair in self.testing_images:
            out = self.model.predict(image_pair[0],verbose=0)
            image = cv.cvtColor(np.squeeze(np.asarray(image_pair[0]).copy()),cv.COLOR_BGR2RGB)
            squeezed_gt = tf.argmax(image_pair[1],axis=-1)
            squeezed_prediction = tf.argmax(out,axis=-1)

            vasc_gt = np.squeeze(image_pair[1][0,:,:,4])
            neural_gt = np.squeeze(image_pair[1][0,:,:,5])
            vasc_pred = np.squeeze(out[0,:,:,4])
            neural_pred = np.squeeze(out[0,:,:,5])

            fig,ax = plt.subplots(1,3)

            ax[0].imshow(image)
            ax[1].imshow(squeezed_gt[0,:,:],vmin=0, vmax=7)
            ax[2].imshow(squeezed_prediction[0,:,:],vmin=0, vmax=7)
            # ax[1].imshow(squeezed_gt[0,:,:]==4)
            # ax[2].imshow(squeezed_prediction[0,:,:]==4)
            plt.show()
            print(np.unique(squeezed_gt[0,:,:]))
            print(np.unique(squeezed_prediction[0,:,:]))


test_images = []
for sample in testing_dataset.take(5):
    #print(sample[0].shape)
    test_images.append([sample[0],sample[1]])
    
# %%

# creating callbacks
sanity_check = SanityCheck(test_images)

def schedule(epoch, lr):
    if (epoch % 3) == 0:
        return(lr*0.7)
    else:
        return(lr)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 mode='min',
                                                 factor=0.8,
                                                 patience=5,
                                                 min_lr=0.000001,
                                                 verbose=True,
                                                 min_delta=0.01,)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('unet_seg_weights.{epoch:02d}-{val_loss:.2f}-{val_precision:.2f}-{val_recall:.2f}.h5',
                                                   save_weights_only=True,
                                                   monitor='loss',
                                                   mode='min',
                                                   verbose=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20,
                                                     monitor='loss',
                                                     mode='min',
                                                     restore_best_weights=True,
                                                     verbose=True,
                                                     min_delta=0.01)

# setting the number of batches to iterate through each epoch to a value much
# lower than what it normaly would be so that we can actually see what is going
# on with the network, as well as have a meaningful early stopping.


# %% fit the network!
# unet.load_weights('./unet_seg_weights.50-0.64-0.93-0.91.h5')
num_steps = 100

weights = {0:0,1:2.72403952,2:2.81034368,3:4.36437716,4:36.66264202, 5:108.40694198, 6:87.39903838}

history = unet.fit(training_dataset,
                   epochs=100,
                   steps_per_epoch=num_steps,
                   validation_data=validation_dataset,
                   class_weight=weights,
                   callbacks=[checkpoint_cb,
                              early_stopping_cb,
                              reduce_lr,
                              sanity_check,])
# %%



# %%
# evaluate the network after loading the weights
# unet.load_weights('./unet_seg_weights.49-0.52-0.94-0.92.h5')
results = unet.evaluate(testing_dataset)
print(results)
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
    plt.imshow(squeezed_gt[0,:,:],vmin=0, vmax=6)
    # print the number of classes in this tile
    print(np.unique(squeezed_gt))
    plt.show()
    # show the flattened predictions
    plt.imshow(squeezed_prediction[0,:,:],vmin=0, vmax=6)
    print(np.unique(squeezed_prediction))
    plt.show()

# %%
# select one of the images cycled through above to investigate further
image_to_investigate = 6

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

# %%
