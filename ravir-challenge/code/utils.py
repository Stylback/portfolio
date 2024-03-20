#-------------------
# Prerequisite modules

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import re
import random

#-------------------

def natural_sort(s):
    
    '''
    This function acts as a sort key for the function "get_file_list"
    Splits a string "s" by the occurrences of pattern "sort", in this case the natural order of numbers.
    
    Parameters:
        s: String. A filename, eg "img_001"
    '''
    
    sort = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(sort, s)]

#-------------------

def get_file_list(main_path, subfolder):
        
    '''
    This function takes a filepath and subfolder and returns a sorted list of filenames from said subfolder
    
    Parameters:
        main_path: String. The absolute path to a directory, eg. "/home/user/Pictures/"
        subfolder: String. The name of a directory, eg. "training_images"
    '''
    
    file_list = []
    data_path = os.path.join(main_path, subfolder)
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            path = os.path.join(root, file)
            file_list.append(path)
    
    file_list.sort(key = natural_sort)
    return file_list

#-------------------

def get_train_val_lists(image_list, mask_list, val_ratio):
        
    '''
    This function takes two lists of filenames and returns them split in training and validation subsets
    
    Parameters:
        image_list: List Object. List containing the filenames of images to split into subsets
        mask_list: List Object. List containing the filenames of masks to split into subsets
        val_ratio: Float between 0 and 1. % of filenames that are placed in the validation subset
    '''
    
    zipped = list(zip(image_list, mask_list))
    random.shuffle(zipped)
    image_list[:], mask_list[:] = zip(*zipped)
    val_split = int(val_ratio * len(image_list))
    
    train_image_list = image_list[val_split:]
    train_mask_list = mask_list[val_split:]
    val_image_list = mask_list[:val_split]
    val_mask_list = image_list[:val_split]

    return train_image_list, train_mask_list, val_image_list, val_mask_list

#-------------------

def load_as_binary(data_list, img_w, img_h, img_c, img_type):
        
    '''
    This function takes a list of filenames and load said files to memory.
    Images have their pixel values normalized to the range [0,1].
    Masks have their pixel values set to either 0 (background) or 1 (segmentation object).
    
    Parameters:
        data_list:
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
        img_type: 'image' or a String. If the image is of type "image" or "mask"
    '''
    
    if img_type == 'image':
        image_data = np.zeros((len(data_list), img_w, img_h, img_c),dtype='float32')
        for i in range(len(data_list)):
            img = cv2.imread(data_list[i], 0)
            img = cv2.resize(img[:,:], (img_w, img_h))
            img = img.reshape(img_w, img_h)/255.
            image_data[i,:,:,0] = img
    
    else:
        image_data = np.zeros((len(data_list), img_w, img_h, img_c),dtype='float32')
        for i in range(len(data_list)):
            img = cv2.imread(data_list[i], 0)
            img = cv2.resize(img[:,:], (img_w, img_h))
            img[img < 255] = 0
            img[img == 255] = 1
            image_data[i,:,:,0] = img
            
    return image_data

#-------------------

def load_as_multiclass(data_list, img_w, img_h, img_c, img_type):
        
    '''
    This function takes a list of filenames and load said files to memory.
    Images have their pixel values normalized to the range [0,1].
    Masks have their pixel values set to either 0 (background) or 1 (classlabel 1) or 2 (classlabel 2).
    
    Parameters:
        data_list:
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
        img_type: 'image' or a String. If the image is of type "image" or "mask"
    '''
    
    if img_type == 'image':
        image_data = np.zeros((len(data_list), img_w, img_h, img_c),dtype='float32')
        for i in range(len(data_list)):
            img = cv2.imread(data_list[i], 0)
            img = cv2.resize(img[:,:], (img_w, img_h))
            img = img.reshape(img_w, img_h)/255.
            image_data[i,:,:,0] = img
    
    else:
        image_data = np.zeros((len(data_list), img_w, img_h, img_c),dtype='float32')
        for i in range(len(data_list)):
            img = cv2.imread(data_list[i], 0)
            img = cv2.resize(img[:,:], (img_w, img_h))
            img[img == 0] = 0
            img[img == 128] = 1
            img[img == 255] = 2
            image_data[i,:,:,0] = img
            
    return image_data
        
#-------------------

def dice_coef(y_true, y_pred):
    
    '''
    Standard Sørensen–Dice coefficient metric
    
    Parameters:
        y_true: Tensor Object. Ground truth
        y_pred: Tensor Object. Data predicted by the model
    '''
    
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#-------------------

def dice_coef_loss(y_true, y_pred):
        
    '''
    Standard Sørensen–Dice coefficient loss function
    
    Parameters:
        y_true: Tensor Object. Ground truth
        y_pred: Tensor Object. Data predicted by the model
    '''
    
    return -dice_coef(y_true, y_pred)

#-------------------

def multi_dice_coef(y_true, y_pred):
    
    '''
    Multiclass Sørensen–Dice coefficient metric
    
    Parameters:
        y_true: Tensor Object. Ground truth
        y_pred: Tensor Object. Data predicted by the model
    '''
    
    smooth = K.epsilon()
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

#-------------------

def multi_dice_coef_loss(y_true, y_pred):
    
    '''
    Multiclass Sørensen–Dice coefficient loss function
    
    Parameters:
        y_true: Tensor Object. Ground truth
        y_pred: Tensor Object. Data predicted by the model
    '''
    
    return -multi_dice_coef(y_true, y_pred)

#-------------------

def combine_generator(image_generator, mask_generator):
        
    '''
    This function combines two "ImageDataGenerator" Objects, ensuring simultaneous "steps" through them. 
    Used in the function "generator".
    
    Parameters:
        image_generator: ImageDataGenerator Object.
        mask_generator: ImageDataGenerator Object.
    '''
    
    while True:
        x = image_generator.next()
        y = mask_generator.next()
        yield(x, y)

#-------------------

def generator(x_train, y_train, batch_size):
        
    '''
    This function takes two lists of images and perform psudo-random image augmentation on each of them.
    
    Parameters:
        x_train: 
        y_train: 
        batch_size: Int larger than 0. The number of images per batch.
    '''
    
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow(x_train,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         seed=1)
    
    mask_generator = image_datagen.flow(y_train,
                                       shuffle=False,
                                       batch_size=batch_size,
                                       seed=1)
   
    train_generator = combine_generator(image_generator, mask_generator)
    
    return train_generator

#-------------------

def plot_model_history(size_x, size_y, title, x_label, y_label, legend, print_keys, model_hist):
        
    '''
    This function creates a line plot over model performance for human evaluation
    
    Parameters:
        size_x: Int larger than 0. Horizontal size of plot
        size_y: Int larger than 0. Vertical size of plot
        title: String. Plot title
        x_label: String. Label of the horizontal x-axis
        y_label: String. Label of the veritcal y-axis
        legend: Boolean. If the legend for evaluation keys should be displayed
        print_keys: Boolean. If evaluation keys should be printed
        model_hist: Model History Object. Performance history of a particular model
    '''
    
    keys = list(model_hist.history.keys())
    if print_keys:
        print("The keys are: ", keys)

    plt.figure(figsize=(size_x, size_y))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    for key in keys:
        plt.plot(model_hist.history[key], label=key)
    
    if legend:
        plt.legend();
    
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()

#--------------------

def get_train_val_list_k_fold(image_list, mask_list, val_ratio, k, fold):
        
    '''
    This function takes two lists of filenames and splits them into training and validation subsets.
    The function utilise the principle of K-fold cross-validation; the dataset is split into k number of "folds", 
    the validation subset is then rotated fold-by-fold to reduce evaluation bias.
    
    Parameters:
        image_list: List Object. List containing the filenames of images to split into subsets
        mask_list: List Object. List containing the filenames of masks to split into subsets
        val_ratio: Float between 0 and 1. % of filenames that are placed in the validation subset
        k: Int. Number of folds
        fold: Int. Which fold-setup to use
    '''
    
    zipped = list(zip(image_list, mask_list))
    image_list[:], mask_list[:] = zip(*zipped)
    
    val_start = int(len(image_list)*(fold-1)/k)
    val_end = int(len(image_list)*fold/k)

    train_image_list = image_list[:val_start] + image_list[val_end:]
    train_mask_list = mask_list[:val_start] + mask_list[val_end:]
    val_mask_list = image_list[val_start:val_end]
    val_image_list = mask_list[val_start:val_end]
    
    return train_image_list, train_mask_list, val_mask_list, val_image_list
    
#--------------------

def save_binary_predictions(predictions):
        
    '''
    This function takes a list of normalized, binary images and 
    converts them to images with pixel values in the range of [0, 255] before saving them to a directory.
    
    Parameters:
        predictions: List Object. A list of images.
    '''
    
    filenames = sorted(os.listdir('/tf/ravir-challenge/dataset/test')) # Change accordingly
    path = '/tf/ravir-challenge/predictions' # Change accordingly
    os.chdir(path)
    index = 0

    for image in predictions:
        image_data = np.zeros((768, 768), dtype='float32')
        for i in range(len(image[:])):
            for j in range(len(image[i,:])):
                pixel_value = image[i,j,:]
                pixel_value = pixel_value * 255
                image_data[i][j] = pixel_value
        
        cv2.imwrite(filenames[index], image_data)
        index = index+1
        print(index, ' out of ', len(predictions), ' converted')    

#--------------------

def save_multiclass_predictions(predictions):
        
    '''
    This function takes a list of normalized, multiclass images and 
    converts them to 2D images with pixel values in the range of [0, 255] before saving them to a directory.
    
    Parameters:
        predictions: List Object. A list of images.
    '''
    
    filenames = sorted(os.listdir('/tf/ravir-challenge/dataset/test')) # Change accordingly
    path = '/tf/ravir-challenge/predictions' # Change accordingly
    os.chdir(path)
    index = 0

    for image in predictions:
        image_data = np.zeros((768, 768), dtype='float32')
        for i in range(len(image[:])):
            for j in range(len(image[i,:])):
                pixel_value = np.argmax(image[i,j,:])
                pixel_value = pixel_value * 128
                image_data[i][j] = pixel_value
        
        cv2.imwrite(filenames[index], image_data)
        index = index+1
        print(index, ' out of ', len(predictions), ' converted')

#-------------------

def get_image_information(data_list, img_w, img_h, img_type, loaded):
        
    '''
    This function takes the first file in a list of filenames and 
    displays image information for testing/debugging purposes.
    
    Parameters:
        data_list: List Object. List containing filenames of images
        img_w: Int larger than 0. Pixel width of image
        img_h: Int larger than 0. Pixel height of image
        img_type: String. What type of image it is, helps keep masks and images apart during debugging
        loaded: Boolean. If the image or mask is already loaded to memory.
    '''
    
    if loaded:
        img_shape = np.shape(data_list[0])
        img_uni = np.unique(data_list[0])
        print('\nShape and Unique values of', img_type, ':\n', img_shape, '\n', img_uni)
        plt.imshow(data_list[0])
        plt.show()
        
    else:
        img = cv2.imread(data_list[0],0)
        img_shape = np.shape(img)
        img_uni = np.unique(img)
        print('\nShape and Unique values of', img_type, ':\n', img_shape, '\n', img_uni)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)