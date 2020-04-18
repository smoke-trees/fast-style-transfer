#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:43:26 2020

@author: tanmay
"""
import cv2
import utils

import tensorflow as tf
import tensorflow_hub as hub


style_path = 'style.jpeg'

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