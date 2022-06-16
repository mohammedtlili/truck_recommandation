from keras.preprocessing.image import ImageDataGenerator
from skimage import io


datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.1,   #% shift
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0,
        horizontal_flip=True,
        fill_mode='reflect', cval=125)    #Also try nearest, constant, reflect, wrap


######################################################################
#Loading a single image for demonstration purposes.
#Using flow method to augment the image

# Loading a sample image
#Can use any library to read images but they need to be in an array form
#If using keras load_img convert it to an array first
m = 'C:/Users/user/Desktop/classes/cam/images - 2022-06-09T112827.736.jpg'
x = io.imread(m);  #Array with shape (256, 256, 3)
print(m, "terminé")
# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images
x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

i = 0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir='C:/Users/user/Desktop/classes/porteur_aug/',
                          save_prefix='aug',
                          save_format='jpg'):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely