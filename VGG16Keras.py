from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, UpSampling2D, Conv2DTranspose, Add, Reshape, concatenate, Concatenate, merge
from keras.optimizers import SGD
import cv2
from skimage.color import rgb2xyz, xyz2luv, rgb2gray, rgb2luv, luv2rgb
from skimage.transform import rescale, resize, downscale_local_mean
import os
import math
from numpy import *
from PIL import Image
from sklearn.cross_validation import train_test_split
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import mean_squared_error

np.set_printoptions(threshold=np.nan)

# input image dimensions
img_rows, img_cols = 224, 224
#batch_size to train
batch_size = 4
# number of epochs to train
nb_epoch = 10

path1 = '/media/chirag/New Volume/Fall 2017/Data Analytics/Project/NewCode/data/train/'    #path of folder of images    
path2 = '/media/chirag/New Volume/Fall 2017/Data Analytics/Project/NewCode/data/bw/'  #path of folder to save images    

listing = os.listdir(path1)
num_samples = size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '/' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L') #need to do some more processing here          
    gray.save(path2 +'/' +  file, "JPEG")
    u = rgb2luv(img)[:,:,1]
    v = rgb2luv(img)[:,:,2]

#Y1 = U.reshape(1, 224, 224, 1)
#Y2 = V.reshape(1, 224, 224, 1)
#print ("Y1.shape: ", Y1.shape)
#print ("Y2.shape: ", Y2.shape)

imlist = os.listdir(path2)

im1 = array(Image.open(path2 + imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

U = array([u.flatten()
              for im2 in imlist],'f')
V = array([v.flatten()
              for im2 in imlist],'f')

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+im2)).flatten()
              for im2 in imlist],'f')
#print immatrix.shape

#X = immatrix
X = np.zeros((224, 224, 3))
#print ("immatrix[1].shape: ", immatrix[1][0])

for j in range(0, 224):
    for k in range(0, 224):
	X[j, k, 0] = immatrix[1][j+k]
	X[j, k, 1] = immatrix[1][j+k]
	X[j, k, 2] = immatrix[1][j+k]

#print ("X.shape: ", X.shape)

immatrix = array([X.flatten()
              for im2 in imlist],'f')

#print ("immatrix.shape: ", immatrix.shape)

X_train, X_test = train_test_split(immatrix, test_size=0.1, random_state=4)
Y1_train, Y1_test = train_test_split(U, test_size=0.1, random_state=4)
Y2_train, Y2_test = train_test_split(V, test_size=0.1, random_state=4)

'''
print ("X_train.shape: ", X_train.shape)
print ("X_test: ", X_test.shape)
print ("Y1_train.shape: ", Y1_train.shape)
print ("Y1_test: ", Y1_test.shape)
print ("Y2_train.shape: ", Y2_train.shape)
print ("Y2_test: ", Y2_test.shape)
'''

#X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
#Y1_train = Y1_train.reshape(Y1_train.shape[0], 1, img_rows, img_cols)
#Y1_test = Y1_test.reshape(Y1_test.shape[0], 1, img_rows, img_cols)
#Y2_train = Y2_train.reshape(Y2_train.shape[0], 1, img_rows, img_cols)
#Y2_test = Y2_test.reshape(Y2_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
print ("X_train.shape: ", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
print ("X_test.shape: ", X_test.shape)
Y1_train = Y1_train.reshape(Y1_train.shape[0], img_rows, img_cols, 1)
print ("Y1_train.shape: ", Y1_train.shape)
Y1_test = Y1_test.reshape(Y1_test.shape[0], img_rows, img_cols, 1)
print ("Y1_test.shape: ", Y1_test.shape)
Y2_train = Y2_train.reshape(Y2_train.shape[0], img_rows, img_cols, 1)
print ("Y2_train.shape: ", Y2_train.shape)
Y2_test = Y2_test.reshape(Y2_test.shape[0], img_rows, img_cols, 1)
print ("Y2_test.shape: ", Y2_test.shape)

'''
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(Y1_train.shape[0], 'train samples')
print(Y1_test.shape[0], 'test samples')
print(Y2_train.sthape[0], 'train samples')
print(Y2_test.shape[0], 'test samples')
'''
#print ("X_train", X_train)

#Model - Layers specification
input_img = Input(shape=(224, 224, 3))
print ("input_img.shape: ", input_img.shape)

conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
print ("conv1.shape: ", conv1.shape)

pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
print ("conv2.shape: ", conv2.shape)

pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
print ("conv3.shape: ", conv3.shape)

upsampledconv3 = UpSampling2D(size=(2, 2))(conv3)
upsampledconv3 = UpSampling2D(size=(2, 2))(upsampledconv3)
print ("upsampledconv3.shape: ", upsampledconv3.shape)

upsampledconv2 = UpSampling2D(size=(2, 2))(conv2)
print ("upsampledconv2.shape: ", upsampledconv2.shape)

#concat = concatenate([conv1,upsampledconv2, upsampledconv3])
#print ("concat.shape: ", concat.shape)

#T matrix of size - 224*224*451
#concat1 = concatenate([input_img,conv1,upsampledconv2, upsampledconv3], axis=0)
concat1 = merge([input_img, conv1, upsampledconv2, upsampledconv3], mode = 'concat', concat_axis=-1)
print ("concat1.shape: ", concat1.shape)

conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(concat1)
conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv4)
print ("conv4.shape: ", conv4.shape)

conv5 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv4)
conv5 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv5)
print ("conv5.shape: ", conv5.shape)

#Q matrix of size - 224*224*144
#concat2 = concatenate([conv4, conv5])
concat2 = merge([conv4, conv5], mode = 'concat', concat_axis=-1)
print ("concat2.shape: ", concat2.shape)

conv6 = Conv2D(32, (3, 3), padding='same', activation='relu')(concat2)
conv6 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv6)
print ("conv6.shape: ", conv6.shape)

conv7 = Conv2D(2, (3, 3), padding='same', activation='relu')(conv6)
conv7 = Conv2D(2, (3, 3), padding='same', activation='relu')(conv7)
print ("conv7.shape: ", conv7.shape)

out_1 = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

out_2 = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

model = Model(inputs=input_img, outputs=[out_1,out_2])

model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')

print model

#Train the neural network
model.fit(X_train, [Y1_train,Y2_train], batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, [Y1_test, Y2_test]))
model.summary()

from keras.models import load_model

model.save('color_model_1080.h5')
#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.2)

image1 = cv2.imread("image_1201.jpg")

imageGray = rgb2gray(1.0/255*image1)

l = rgb2luv(image1)[:,:,0]

image1 = resize(image1,(224,224))

image1 = image.img_to_array(image1)

#L Channel

#print X

#X = immatrix
X = np.zeros((224, 224, 3))

image1 = image1.flatten()

for j in range(0, 224):
    for k in range(0, 224):
	X[j, k, 0] = imageGray[1][j+k]
	X[j, k, 1] = imageGray[1][j+k]
	X[j, k, 2] = imageGray[1][j+k]

X = X.reshape(1, 224, 224, 3)

u, v = model.predict(X);

canvas = np.empty((224, 224, 3))

canvas[:,:,0] = l
canvas[:,:,1] = u
canvas[:,:,2] = v
imsave("img_luv_result.png", canvas)
imsave("img_result.png", luv2rgb(canvas))
imsave("img_gray_scale.png", rgb2gray(luv2rgb(canvas)))

