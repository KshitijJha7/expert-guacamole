
import os
import tensorflow as tf
from keras.models import Model
import tensorflow
import numpy as np

import cv2


i_shape = 224

input_set = np.zeros((1,224,224, 3))
#gender_set = np.zeros((1, 1))
#age_set = np.empty((self.batch_size,1))
gender_dict = {'F':0, 'M' : 1}

filename = './images/7_18_F.jpg'

img = cv2.imread(filename)
img = cv2.resize(img,(i_shape,i_shape))
input_set[0,] = img

new_model = tf.keras.models.load_model('custom_model.h5')

# Show the model architecture
new_model.summary()


result = new_model.predict(input_set)
