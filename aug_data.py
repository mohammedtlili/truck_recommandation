from keras.preprocessing.image import ImageDataGenerator
from skimage import io


datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0,
        horizontal_flip=True,
        fill_mode='reflect', cval=125)    #Also try nearest, constant, reflect, wrap

####################################################################
# Multiple images.
# Manually read each image and create an array to be supplied to datagen via flow method
dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = 'C:/Users/user/Desktop/classes/por/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
        if (image_name.split('.')[1] == 'jpg'):
                image = io.imread(image_directory + image_name)
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                dataset.append(np.array(image))

x = np.array(dataset)




# Let us save images to get a feel for the augmented images.
# Create an iterator either by using image dataset in memory (using flow() function)
# or by using image dataset from a directory (using flow_from_directory)
# from directory can beuseful if subdirectories are organized by class

# Generating and saving 10 augmented samples
# using the above defined parameters.
# Again, flow generates batches of randomly augmented images

i = 0
for batch in datagen.flow(x, batch_size=200,
                          save_to_dir='C:/Users/user/Desktop/classes/porteur_aug/',
                          save_prefix='aug',
                          save_format='jpg'):
        i += 1
        if i > 10:
                break  # otherwise the generator would loop indefinitely