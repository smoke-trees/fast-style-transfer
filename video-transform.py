#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:43:26 2020

@author: tanmay
"""
import os
import cv2
import utils

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)


style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

style_image = utils.load_img(style_path)

utils.imshow(style_image, 'Style Image')
 
# Opens the Video file
cap = cv2.VideoCapture('test.mp4')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('images_transformed/' + str(i) + '.jpg', frame)
    i += 1
 
cap.release()
cv2.destroyAllWindows()

imgs = sorted(os.listdir('images_dissected/'))

for img in imgs:
    content_image = utils.load_img('images_dissected/' + img)
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    copy = utils.tensor_to_image(stylized_image)
    copy.save('images_transformed/transformed_' + img)
    
    
imgs_transformed = sorted(os.listdir('images_transformed/'))

img_array = []
for img in imgs_transformed:
    image = cv2.imread('images_transformed/' + img)
    height, width, layers = image.shape
    size = (width,height)
    img_array.append(image)
 
out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

# Write the file 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()