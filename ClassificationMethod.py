import tensorflow as tf
import numpy as np
from numpy import ndarray
import skimage.data as data
from skimage.viewer import ImageViewer
import skimage.color as colr
from skimage.color import rgb2xyz, xyz2luv, rgb2gray, rgb2luv, luv2rgb
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Conv2DTranspose, Add
from keras import losses
from scipy.misc import imsave

#import nn_layers
#import os
#import cellDataClass as dataClass
#import preprocess_images as procIm
np.set_printoptions(threshold=np.nan)

#Get Images
img = cv2.imread("index.png")

img = resize(img,(224,224))

# Import map images into the CIELUV colorspace
#img_xyz = rgb2xyz(img)
#imsave("img_luv.png", rgb2luv(img))

#Lumninace Channel - black and white image
#X = xyz2luv(1.0/255*img_xyz)[:,:,0]
X = rgb2luv(1.0/255*img)[:,:,0]
#UV Channel
#Y = xyz2luv(1.0/255*img_xyz)[:,:,1:]
Y = rgb2luv(1.0/255*img)[:,:,1:]
#print Y

# Since U and V values lies between -100 and 100.
Y = Y / 100

X = X.reshape(1, 224, 224, 1)
Y = Y.reshape(1, 224, 224, 2)

print X.shape
print Y.shape

#Building the neural network
input_img = Input(shape=(224, 224, 1))

#input_img = tf.reshape(input_img, [1, 224, 224, 1])
print input_img.shape

conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
print ("conv1.shape: ", conv1.shape)

conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
print ("conv2.shape: ", conv2.shape)

pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv2)
print ("pool1.shape: ", pool1.shape)

conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
print ("conv3.shape: ", conv3.shape)

conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
print ("conv4.shape: ", conv4.shape)

pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv4)
print ("pool2.shape: ", pool2.shape)

conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
print ("conv5.shape: ", conv5.shape)

conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
print ("conv6.shape: ", conv6.shape)

conv7 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv6)
print ("conv7.shape: ", conv7.shape)

pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv7)
print ("pool3.shape: ", pool3.shape)

conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
print ("conv8.shape: ", conv8.shape)

conv9 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv8)
print ("conv9.shape: ", conv9.shape)

conv10 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv9)
print ("conv10.shape: ", conv10.shape)

print "after conv10"

#Right-hand side layers
convr3 = Conv2DTranspose(256, (1, 1), padding='valid', activation='relu')(conv10)
print ("convr3.shape: ", convr3.shape)

convr3Upsampled = UpSampling2D(size=(2, 2), data_format=None)(convr3)
print ("convr3Upsampled.shape: ", convr3Upsampled.shape)

#elem2 = tf.add(conv7,convr3Upsampled,'elem2')
elem2 = Add()([conv7, convr3Upsampled])
print ("elem2.shape: ", elem2.shape)

convr2 = Conv2DTranspose(128, (1, 1), padding='same', activation='relu')(elem2)
print ("convr3.shape: ", convr3.shape)

convr2Upsampled = UpSampling2D(size=(2, 2), data_format=None)(convr2)
print ("convr2Upsampled.shape: ", convr2Upsampled.shape)

#elem1 = tf.add(conv4,convr2Upsampled,'elem1')
elem1 = Add()([conv4, convr2Upsampled])
print ("elem1.shape: ", elem1.shape)

convr1 = Conv2DTranspose(64, (1, 1), padding='same', activation='relu')(elem1)
print ("convr1.shape: ", convr1.shape)

convr1Upsampled = UpSampling2D(size=(2, 2), data_format=None)(convr1)
print ("convr1Upsampled.shape: ", convr2Upsampled.shape)

#elem0 = tf.add(conv2,convr1Upsampled,'elem0')
elem0 = Add()([conv2, convr1Upsampled])
print ("elem0.shape: ", elem0.shape)

pred = Conv2D(2, (3, 3), padding='same', activation='softmax')(elem0)
print ("pred.shape: ", pred.shape)

#Finish Model
#predictions = Dense(50, activation='softmax')(convb3)
model = Model(inputs=input_img, outputs=pred)
model.compile(optimizer='adam',loss='mse')

#Train the neural network
model.fit(x=X, y=Y, batch_size=1, epochs=10)
print(model.evaluate(X, Y, batch_size=1))

# Output colorizations
output = model.predict(X)
output = output / 100
#print ("output: ", output)
canvas = np.empty((224, 224, 3))
canvas[:,:,0] = X[0][:,:,0]
canvas[:,:,1:] = output[0]
imsave("img_luv_result.png", canvas)
imsave("img_result_10.png", rgb2luv(canvas))
imsave("img_gray_scale.png", rgb2gray(luv2rgb(canvas)))