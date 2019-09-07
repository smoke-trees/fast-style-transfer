# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:42:30 2019

@author: tanma
"""

from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

from style_transfer_1 import VGG16_AvgPool, unpreprocess, scale_img
from skimage.transform import resize
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K



def gram_matrix(img):
  X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
  
  G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
  return G


def style_loss(y, t):
  return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def minimize(fn, epochs, batch_shape):
  t0 = datetime.now()
  losses = []
  x = np.random.randn(np.prod(batch_shape))
  for i in range(epochs):
    x, l, _ = fmin_l_bfgs_b(
      func=fn,
      x0=x,
      maxfun=20
    )
    x = np.clip(x, -127, 127)
    print("iter=%s, loss=%s" % (i, l))
    losses.append(l)

  print("duration:", datetime.now() - t0)
  plt.plot(losses)
  plt.show()

  newimg = x.reshape(*batch_shape)
  final_img = unpreprocess(newimg)
  return final_img[0]


if __name__ == '__main__':
  path = 'content.jpeg'
  img = image.load_img(path)
  x = image.img_to_array(img)

  x = np.expand_dims(x, axis=0)

  x = preprocess_input(x)

  batch_shape = x.shape
  shape = x.shape[1:]

  vgg = VGG16_AvgPool(shape)

  symbolic_conv_outputs = [
    layer.get_output_at(1) for layer in vgg.layers \
    if layer.name.endswith('conv1')
  ]

  multi_output_model = Model(vgg.input, symbolic_conv_outputs)

  style_layers_outputs = [K.variable(y) for y in multi_output_model.predict(x)]

  loss = 0
  for symbolic, actual in zip(symbolic_conv_outputs, style_layers_outputs):
    loss += style_loss(symbolic[0], actual[0])

  grads = K.gradients(loss, multi_output_model.input)

  get_loss_and_grads = K.function(
    inputs=[multi_output_model.input],
    outputs=[loss] + grads
  )


  def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


  final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
  plt.imshow(scale_img(final_img))
  plt.show()
