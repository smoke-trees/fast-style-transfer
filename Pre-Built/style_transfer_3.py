# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:47:15 2019

@author: tanma
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from style_transfer_1 import VGG16_AvgPool, VGG16_AvgPool_CutOff, unpreprocess, scale_img
from style_transfer_2 import gram_matrix, style_loss, minimize
from scipy.optimize import fmin_l_bfgs_b

def load_img_and_preprocess(path, shape=None):
  img = image.load_img(path, target_size=shape)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  return x

content_img = load_img_and_preprocess(
  'style.jpg',
)

h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess(
  'content.jpeg',
  (h, w)
)

batch_shape = content_img.shape
shape = content_img.shape[1:]

vgg = VGG16_AvgPool(shape)

content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
content_target = K.variable(content_model.predict(content_img))

symbolic_conv_outputs = [
  layer.get_output_at(1) for layer in vgg.layers \
  if layer.name.endswith('conv1')
]

style_model = Model(vgg.input, symbolic_conv_outputs)

style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

style_weights = [0.2,0.4,0.3,0.5,0.2]

loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
  loss += w * style_loss(symbolic[0], actual[0])

grads = K.gradients(loss, vgg.input)

get_loss_and_grads = K.function(
  inputs=[vgg.input],
  outputs=[loss] + grads
)


def get_loss_and_grads_wrapper(x_vec):
  l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
  return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()
