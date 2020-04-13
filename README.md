```python
import utils

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
```


```python
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
```


```python
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
```


```python
content_image = utils.load_img(content_path)
style_image = utils.load_img(style_path)

plt.subplot(1, 2, 1)
utils.imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
utils.imshow(style_image, 'Style Image')
```


![png](https://github.com/smoke-trees/Anime_Neural_Style_Transfer/blob/master/images/output_3_0.png)


## Fast Style Transfer using TF-Hub

Module built on the original paper


```python
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
utils.tensor_to_image(stylized_image)
```




![png](https://github.com/smoke-trees/Anime_Neural_Style_Transfer/blob/master/images/output_5_0.png)




```python

```
