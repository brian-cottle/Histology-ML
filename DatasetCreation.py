# %% Importing packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import patches
plt.rcParams['figure.figsize'] = [5, 10]
import os
import tensorflow as tf
from skimage import measure
from skimage import draw
from scipy.spatial import distance
import time
from joblib import Parallel, delayed
from natsort import natsorted

# %% Defining Functions
#############################################################
#############################################################

def load_image_names(folder):
    '''This function reads the images within a folder while filtering 
       out the weird invisible files that macos includes in their folders'''
    
    file_list = []
    for file_name in os.listdir(folder):

        # check if the first character of the name is a '.', skip if so
        if file_name[0] != '.': 
            file_list.append(file_name)

    return(file_list)
        

#############################################################

def get_bounding_boxes(binary_image):
    '''This function receives a binary image, and returns a list of the
       bounding boxes that surround the positive connected components'''

    labeled_image = measure.label(binary_image) # labeling image
    regions = measure.regionprops(labeled_image) # getting region props
    
    bboxes = [] # an appended list for the bounding box coordinates

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

        # note that the below bounding box format is different than the format
        # that is used in the storage within tfrecord files. The record files
        # store the x, y, width, and height in their own respective lists, 
        # instead of having a list of lists.
        bboxes.append([x,y,width,height])

    return(bboxes)
    

#############################################################

def random_center_near_outline(outline,
                               x_bounds,
                               y_bounds, 
                               class_seg=False, 
                               tile_size = 1024,
                               sample_center_xy = [0,0]):
    '''this function receives a binary outline (in the specific case of a tissue 
       sample) as well as the x and y limits of that outline. Returns a random
       x y pair that is within the boundaries of that outline.'''

    if not class_seg:
        # keep iterating until you get a random number fulfilling the 
        # requirements    
        while True:
            
            # getting the range of x and y values within which the random number
            # should be generated
            max_x_range = x_bounds[1]-x_bounds[0]
            max_y_range = y_bounds[1]-y_bounds[0]
            
            # produces a random number between the bounds provided
            random_x_center = int(np.floor(np.random.random() * max_x_range + 
                                  x_bounds[0]))
            random_y_center = int(np.floor(np.random.random() * max_y_range + 
                                  y_bounds[0]))

            # if the pair generated is within the outine, leave the function
            if outline[random_x_center,random_y_center]:
                break
    
    else:
        
        x_width = (x_bounds[1] - x_bounds[0])
        y_width = (y_bounds[1] - y_bounds[0])


        max_x_range = tile_size - x_width
        max_y_range = tile_size - y_width
        random_x_center = int(np.floor(np.random.random() * max_x_range + 
                              sample_center_xy[0] - max_x_range/2))
        random_y_center = int(np.floor(np.random.random() * max_y_range + 
                              sample_center_xy[1] - max_y_range/2))


    return(random_x_center,random_y_center)

#############################################################

def get_subsampling_coordinates(image, 
                                num_samples=50,
                                tile_size=1024,
                                persistence=1000):
    '''This function receives an image with segmentations as well as a user 
       determined number of samples to take from the image (default 50, though 
       that number usually isn't reached). The size of the tiles sub-sampled as
       well as how many times the function should try to sample the image 
       randomly are also able to be user set.'''

    # seemingly fastest way to get the inverse of the outline of the image, or 
    # a filled binary segmentation of the tissue.
    outline = image[:,:,3] > 1
    outline_props = measure.regionprops(outline.astype(np.uint8))
    # retrieving the x and y limits of the outline through the bounding box
    outline_bbox = outline_props[0].bbox

    # extracting the min and max x and y values
    x_min_max = [outline_bbox[0],outline_bbox[2]]
    y_min_max = [outline_bbox[1],outline_bbox[3]]

    # initializing list for the sample box centers 
    random_centers = []
    # binary variable for permitting a random subsampling center to be saved
    norm_test = 1
    # counter used for terminating the while loop, as a bit of a lazy solution 
    # to a bug I didn't really want to thoroughly investigate. Could use for 
    # loop pretty easily in the future too.
    Counter = 0

    while True:
        # retrieve a new random box center
        new_center = random_center_near_outline(outline,x_min_max,y_min_max)
        
        # if this isn't the first random center generated, proceed to center
        # distance checking
        if len(random_centers)>0:
            # make sure norm_test is 1 to begin with
            norm_test = 1

            # iterate through all the previously saved random centers
            for c in random_centers:
                # currently, the min distance between samples is tile_size, 
                # which allows for some overlap, but not very much in practice
                min_distance = tile_size

                # calculate the euclidean norm distance between the current
                # random center "c" and the potentially new random center
                euclidean_norm = distance.euclidean(c,new_center)
                # if any of the distances between the new center and old ones is
                # smaller than min_distance, norm_test=0, which doesn't allow
                # that new center to be saved
                if euclidean_norm < min_distance:
                    norm_test = 0
            
            # if norm_test made it through all the already saved random centers
            # without being zero, the center is then added to the list
            if norm_test:
                random_centers.append(new_center)
        
        # from above, if this is the first center proposed, save it anyway
        elif len(random_centers)==0:
            random_centers.append(new_center)
        
        # increment the number of attempts
        Counter+=1
        
        # if we have collected as many samples as were asked for, leave
        if len(random_centers) >= num_samples:
            break
        
        # if we hit the number of attempts allowed, leave
        if Counter >= persistence:
            break

    # returns tile size as well as the centers, and min and max values as they 
    # are useful to know down the line for further processing
    return(tile_size, random_centers, x_min_max, y_min_max)

#############################################################

def get_subsampling_coordinates_classfocused(image,
                                             class_id=5,
                                             num_samples=10,
                                             tile_size=1024,
                                             persistence=1000):
    '''This function receives an image and returns several pseudo-random 
       tile centers that include an instance or object that is classified as
       the class "class_id". The persistence is the number of times the function
       will "draw" a random location within the boundaries of the tissue in an 
       attempt to place a random tile somewhere that includes the target 
       class.'''

    segmentation = image[:,:,3] == class_id
    bboxes = get_bounding_boxes(segmentation)

    # initializing list for the sample box centers 
    random_centers = []
    # binary variable for permitting a random subsampling center to be saved
    norm_test = 1

    # using that for loop that should have been implemented in 
    # get_subsampling_coordinates
    for idx in range(persistence):
        for box in bboxes:

            # getting the min and max that still contain the full bounding
            # box for the segmentation
            x_min_max = [int(box[0]-np.floor(box[2]/2)),
                         int(box[0]+np.floor(box[2]/2))]

            y_min_max = [int(box[1]-np.floor(box[3]/2)),
                         int(box[1]+np.floor(box[3]/2))]

            # retrieve a new random box center, but this time using the
            # "class_seg" option
            new_center = random_center_near_outline(
                segmentation,x_min_max,
                y_min_max, class_seg=True,
                tile_size=tile_size,
                sample_center_xy=[box[0],box[1]]
                )

            # if this isn't the first random center generated, proceed to center
            # distance checking
            if len(random_centers)>0:
                # make sure norm_test is 1 to begin with
                norm_test = 1

                # iterate through all the previously saved random centers
                for c in random_centers:
                    # currently, the min distance between samples is tile_size, 
                    # which allows for some overlap, but not very much in 
                    # practice
                    min_distance = tile_size

                    # calculate the euclidean norm distance between the current
                    # random center "c" and the potentially new random center
                    euclidean_norm = distance.euclidean(c,new_center)

                    # if any of the distances between the new center and old 
                    # ones is smaller than min_distance, norm_test=0, which 
                    # doesn't allow that new center to be saved
                    if euclidean_norm < min_distance:
                        norm_test = 0
                
                # if norm_test made it through all the already saved random 
                # centers without being zero, the center is then added to the
                # list
                if norm_test:
                    random_centers.append(new_center)
            
            # from above, if this is the first center proposed, save it anyway
            elif len(random_centers)==0:
                random_centers.append(new_center)
            
        # if we have collected as many samples as were asked for, leave
        if len(random_centers) >= num_samples:
            break

    # returns tile size as well as the centers, and min and max values as they 
    # are useful to know down the line for further processing
    return(tile_size, random_centers)

#############################################################

def show_tiled_samples(image, centers, tile_size=1024,seg=False):
    '''this function receives an image as well as the centers variable returned 
       by get_subsampling_coordinates, and creates a visualization of where
       samples are being taken from the image provided image.'''

    # extract the centers
    xy_centers = centers[1]

    # extract the bounds
    x_bounds = centers[2]
    y_bounds = centers[3]

    # getting the width from the center of the bounding box
    half_dimension = np.floor(tile_size / 2)

    # list for the locations within the image for visualization
    box_locations = []

    # get the bottom left corner of the image for visualization with 
    # matplotlib patches below
    for xy in xy_centers:
        box_locations.append([xy[1]-half_dimension-y_bounds[0],
                              xy[0]-half_dimension-x_bounds[0]])

    # crop the monstrously huge images to be just the tissue for visualization
    print(x_bounds)
    print(y_bounds)
    cropped_image = image[x_bounds[0]:x_bounds[1],
                        y_bounds[0]:y_bounds[1],0:3]

    seg_image = image[x_bounds[0]:x_bounds[1],
                        y_bounds[0]:y_bounds[1],3]

    # subplots so we can access the ax object
    fig, ax = plt.subplots()

    # show the image
    if not seg:
        ax.imshow(cropped_image)
    else:
        ax.imshow(seg_image)

    # for each sampled box, create a rectangle and add it to the image
    for locations in box_locations:
        p = patches.Rectangle(locations,tile_size,tile_size, edgecolor='r',
                              facecolor='none', linewidth=1)
        ax.add_patch(p)
    
    # print how many boxes there were in the sub-sample set
    print(f'found {len(box_locations)} successful samples.')

    return()


#############################################################

def save_image_slices(image, 
                      image_name,
                      centers,
                      class_correction=0,
                      class_id=2):
    '''this function receives an image with segmentations, that image's name, 
       and the list of centers that were provided and vetted using 
       get_subsampling_coordinates. The image is then cropped to create each
       sub sampled image, and they are saved in a new directory in the parent 
       folder of the dataset_directory.'''


    # extract variables from the centers object
    tile_size = centers[0]
    half_size = np.floor(tile_size / 2)
    random_centers = centers[1]

    current_dir = os.getcwd()
    os.chdir('./..')

    # create directory with the date it was produced
    new_dir = './sub_sampled_'+time.strftime('%Y%m%d')
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    os.chdir(new_dir)

    for idx,xy in enumerate(random_centers):

        # Get the index values that should be used for generating the images
        xmin = int(xy[0]-half_size)
        xmax = int(xy[0]+half_size)
        ymin = int(xy[1]-half_size)
        ymax = int(xy[1]+half_size)

        # this can be used for troubleshooting
        # print(f'x1: {xmin}, x2: {xmax}, y1: {ymin}, y2: {ymax}')
        
        # crop!
        current_crop = image[xmin:xmax,ymin:ymax]


        # implementing correcting for a missing or irrelevant class in the
        # current dataset, for example as written now removes all references
        # to neural tissue in the segmentation, if you set class_correction = 5
        if class_correction > 0:
            seg = current_crop[:,:,3]
            crop_class_mask = seg==class_correction
            seg[crop_class_mask] = class_correction - 1
            current_crop[:,:,3] = seg

        # accounting for the a jump in the background to first segmentation
        seg = current_crop[:,:,3]
        seg_zero_mask = seg==0
        seg[seg_zero_mask] = 1

        # write the file name, appending the sub-sampled number to the original
        cv.imwrite(
            image_name[:-4] + 
            f'_class_{class_id}_subsampled_{idx}.png',
            current_crop
            )

    os.chdir(current_dir)

    return()

#############################################################

def double_check_produced_dataset(new_directory,image_idx=0):
    '''this function samples a random image from a given directory, crops off 
       the ground truth from the 4th layer, and displays the color image to 
       verify they work.'''
    os.chdir(new_directory)
    file_names = load_image_names(new_directory)
    file_names = natsorted(file_names)
    # pick a random image index number
    if image_idx == 0:
    image_idx = int(np.random.random()*len(file_names))
    else:
        pass

    print(image_idx)
    # reading specific file from the random index
    tile = cv.imread(file_names[image_idx],cv.IMREAD_UNCHANGED)
    # changing the color for the tile from BGR to RGB
    color_tile = cv.cvtColor(tile[:,:,0:3],cv.COLOR_BGR2RGB)
    fig, (ax1,ax2) = plt.subplots(1,2)
    print(file_names[image_idx])
    # plotting the images next to each other
    ax1.imshow(color_tile)
    ax2.imshow(tile[:,:,3],vmin=0, vmax=6)
    print(np.unique(tile[:,:,3]))
    plt.show()

#############################################################

def joblib_parallel_function_class_focused(file,
                                           class_id=5,
                                           num_samples=200,
                                           tile_size=1024,
                                           class_correction=0):
    
    '''Put together to run all the necessary functions above in a parallel loop
       using joblib to create the dataset significantly faster than in 
       serial.'''
    
    # load the current image file
    image = cv.imread(file,cv.IMREAD_UNCHANGED)
    # pad the image to prevent sections from going outside the image bounds
    image = cv.copyMakeBorder(image,1000,1000,1000,1000,cv.BORDER_REPLICATE)
    # run either of the get_subsampling_coordinates functions
    centers = get_subsampling_coordinates_classfocused(image,
                                                       class_id=class_id,
                                                       num_samples=num_samples, 
                                                       tile_size=tile_size)
    # save the image segmentations that were found from the previous function
    # also added class correction for this dataset generation, should be 
    # changed in the future
    save_image_slices(image, file, centers,class_correction)
    return()

#############################################################
#############################################################

# %% Reading the contents of the dataset directory

# Current directory is on separate hard drive
dataset_directory = ('/home/briancottle/Research/'
                     'Semantic_Segmentation/ML Dataset 2')
os.chdir(dataset_directory)

# %% initializing variables
num_samples = 200
tile_size = 1024

# load image names from within dataset directory
file_names = load_image_names(dataset_directory)

# %%
contains_names_vascular = Parallel(
    n_jobs=8, verbose=5)(delayed(joblib_parallel_function_class_focused)
    (name,
     class_id=2,
     num_samples=200,
     tile_size=1024,
     class_correction=0) for name in file_names
    )
# %%

double_check_produced_dataset('/home/briancottle/Research/'
                              'Semantic_Segmentation/sub_sampled_20220615')

# %%
