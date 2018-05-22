
# coding: utf-8

# In[1]:

import os
path = 'C:/Users/mehul/Desktop/data'
name = os.listdir(path)
classes = []
for n in name:
    temp = n.split('_')
    temp[-1] = temp[-1].split('.')[0]
    temp = temp[3:]
    classes.append(temp)
del(temp)


# In[2]:

labels = set()
for i in classes:
    for j in i:
        labels.add(int(j))
labels = list(labels)


# In[3]:

from keras.utils import np_utils
import numpy as np


# In[11]:

y_train = np.zeros((1727,76))
for i in range(len(classes)):
    for j in classes[i]:
        y_train[i,labels.index(int(j))] = 1
Y_train = y_train.astype('int')
num_classes = 76


# In[5]:

import cv2
X_train = []
for n in os.listdir(path):
    p = path + '/' + n
    img  = cv2.imread(p,0)
    X_train.append(img)
X_train = np.array(X_train)
X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(1727,128,128,1)
X_train.shape
#del(x_train)


# In[9]:

from keras.layers.normalization import BatchNormalization
from keras import backend as K
K.set_learning_phase(1) #set learning phase
batch_size = 100
num_epochs = 2
kernel_size = 3  
pool_size = 2
conv_depth_0 = 64
conv_depth_1 = 80 
conv_depth_2 = 100
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512


# In[10]:

from keras.models import Model 
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy
inp = Input(shape=(128, 128, 1)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_0, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_0, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
batch = BatchNormalization(axis = -1)(conv_2)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size),strides=2)(batch)

conv_3 = Convolution2D(conv_depth_1,(kernel_size,kernel_size), padding='same' , activation = 'relu')(pool_1)
conv_4 = Convolution2D(conv_depth_1,(kernel_size,kernel_size), padding='same' , activation = 'relu')(conv_3)
batch1 = BatchNormalization(axis = -1)(conv_4)

pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size),strides=2)(batch)
drop_1 = Dropout(drop_prob_1)(pool_2)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_5 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_5)
drop_2 = Dropout(drop_prob_1)(pool_3)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='sigmoid')(drop_3)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='binary_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy
model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation


# In[30]:

import heapq
data = [0.5, 0.7, 0.3, 0.3, 0.3, 0.4, 0.5]
heapq.nlargest(3, data)


# In[27]:

from keras.models import load_model
import json
#model.save('my_model.h5')  
#del model
from keras.models import model_from_json

model = load_model('my_model.h5')

json_file = open('n_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("n_model.h5")
model1 = loaded_model
print("Loaded n model)")


# In[43]:

a = model.output
print(a)
pred = model.predict(X_train,verbose = 1)
pred1 = model1.predict(X_train,verbose = 1)


# In[67]:

y1 = []
for i in range(len(classes)):
    k = len(classes[i])
    if(k>3):
        print(heapq.nlargest(k,pred[i]))
    a = heapq.nlargest(k,pred[i])
    y1.append([k,a])


# In[68]:

y_train1 = np.zeros((1727,3))
for i in range(len(y1)):
    a = y1[i][0]
    y_train1[i,(min(a-1,2))] = 1
Y_train1 = y_train1.astype('int')
num_classes1 = 3


# In[74]:

for i in range(110,200):
    temp = np.argsort(pred1[i])
    print('temp',temp[-1]+1)
    temp1 = np.argsort(Y_train1[i])
    print('temp1',temp1[-1]+1)
    a = np.argsort(pred[i])
    a = a[-(temp[-1]+1):]
    print(a)
    #print(pred[i])
    print(a)
    for i in range(len(a)):
        print(labels[a[i]])
    n = os.listdir(path)
    print(n[i])
    print()
    print()


# In[20]:

from keras import backend as K
K.set_learning_phase(1) #set learning phase
batch_size = 100
num_epochs = 20
kernel_size = 3  
pool_size = 2
conv_depth_1 = 64 
conv_depth_2 = 80
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512


# In[21]:

from keras.models import Model 
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy
inp = Input(shape=(128, 128, 1)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(100, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes1, activation='softmax')(drop_3)

model1 = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model1.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model1.fit(X_train, Y_train1,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation


# In[19]:

from keras.models import load_model

#model.save('my_model.h5')  
#del model

model1 = load_model('n_model.h5')


# In[ ]:



