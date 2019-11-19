#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#tf.config.allow_growth = True
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config....)
from tensorflow import keras


# In[5]:


data = keras.datasets.fashion_mnist

(train_X, train_y), (test_X,test_y) = data.load_data()

class_names = ['t-shirt', 'trouser', 'pullover', 'dress'
                ,'coat', 'sandal', 'shirt', 'sneaker'
                , 'bag', 'ankle boot']

train_X = train_X/255
test_X = test_X/255


# In[7]:


plt.imshow(train_X[7], cmap= 'binary')


# In[ ]:


def convolve(image,fltr):
    r_p = 0
    c_p = 0
    conv_list = []
    while (r_p+1) <= image.shape[0]-1 :
        while (c_p+1) <= image.shape[1]-1 :
            x = np.sum(np.multiply(image[r_p : r_p+2 , c_p : c_p+2],fltr))
            conv_list.append(x)
            c_p += 1
        r_p += 1
        c_p = 0
    return conv_list
img_matrix = np.array(train.iloc[6,1:]).reshape(28,28)
flt = np.matrix([[1,1],[0,0]])
conv = np.array(convolve(img_matrix,flt)).reshape(27,27)
plt.imshow(img_matrix, cmap='gray')
plt.show()
plt.imshow(conv, cmap='gray')
plt.show()


# In[33]:


with tf.device('GPU:0'):
    model = keras.Sequential([ 
        #keras.layers.Conv2D(filters=32 ,kernel_size=3, activation='relu',input_shape=(28,28,1)),
        keras.layers.Flatten(input_shape=(28,28)),
        #keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2560, activation='relu'),
        keras.layers.Dense(2560, activation='relu'),
        #keras.layers.Dense(2560, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
        ])
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    import time
    tic = time.time()
    from warnings import filterwarnings
    filterwarnings
    model.fit(train_X, train_y,batch_size=1024, epochs=3)
    toc = time.time()
    print('time : {:0.1f} sec '.format(toc-tic))


# In[72]:


#predictions
train_loss, train_accuracy = model.evaluate(train_X, train_y,verbose=False )
test_loss, test_accuracy = model.evaluate(test_X, test_y, verbose = False )


# In[73]:


print('trin_accuracy : {}'.format(train_accuracy))
print('test_accuracy : {}'.format(test_accuracy))


# In[74]:


predictions = model.predict(test_X)


# In[76]:


plt.imshow(test_X[26], cmap='binary')
plt.title(class_names[test_y[26]])

