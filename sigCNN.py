#!/usr/bin/env python
# coding: utf-8

# In[1]:


#start with checking datasets
import os
os.getcwd()


# In[3]:


#Changing path to datsets
os.chdir("Datasets\BHSig260")
os.getcwd()


# In[4]:


path_hindi = "./BHSig260/Hindi/"


# In[5]:


# Get the list of all directories and sort them
dir_list = next(os.walk(path_hindi))[1]
dir_list.sort()


# In[6]:


data_groups = []
for directory in dir_list:
    images = os.listdir(path_hindi+directory)
    images.sort()
    images = [path_hindi+directory+'/'+x for x in images]
    data_groups.append(images)


# In[7]:


#how signatures images will be in data_groups
data_groups[:2]


# In[8]:


path_bengali = "./BHSig260/Bengali/"


# In[9]:


dir_list = next(os.walk(path_bengali))[1]
dir_list.sort()


# In[10]:


for directory in dir_list:
    images = os.listdir(path_bengali+directory)
    images.sort()
    images = [path_bengali+directory+'/'+x for x in images]
    data_groups.append(images)


# In[11]:


#Now all the images from both Hindi and Bengali both the genuine and forged added to data groupes


# In[12]:


# Quick check to confirm we have data of all the 160 individuals, Hindi 160, Bengali 100
len(data_groups)


# In[13]:


data_lengths = [len(x) for x in data_groups]
# Quick check to confirm that there are 24 Genuine signatures for each individual and 30 forged so total 54
print(data_lengths)


# In[14]:


"""Train-Validation-Test Split
Signatures of 120 people are used for training
Signatures of 20 people are used for validation
Signatures of 20 people are used for testing"""


# In[18]:


#approx to 2:8 ratio that gives better result as per research paper analysis
from sklearn.model_selection import train_test_split
train, test = train_test_split(data_groups, test_size=50/260, random_state=1)
train, val = train_test_split(train, test_size=40/210, random_state=1)


# In[19]:


#train, test and validation size
train_size = len(train)
test_size = len(test)
val_size = len(val)
print(len(train), len(val), len(test))


# In[20]:


# Delete unnecessary variables
del data_groups


# In[21]:


# All the images will be converted to the same size before processing (from research paper)
img_h, img_w = 64, 128


# In[22]:


#modules needed for plotting, data preprocessing and for other operations 
import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import time
import itertools
import random

from sklearn.utils import shuffle


# In[23]:


def visualize_sample_signature():
    '''Function to randomly select a signature from train set and
    print two genuine copies and one forged copy'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
    k = np.random.randint(len(train))
    img_names = random.sample(train[k], 2)
    label1 = 0 if "F" in img_names[0] else 1
    label2 = 0 if "F" in img_names[1] else 1
    img1 = cv2.imread(img_names[0], 0)
    img2 = cv2.imread(img_names[1], 0)
    img1 = cv2.resize(img1, (img_w, img_h))
    img2 = cv2.resize(img2, (img_w, img_h))

    ax1.imshow(img1, cmap = 'gray')
    ax2.imshow(img2, cmap = 'gray')

    ax1.set_title('Genuine Copy' if label1 == 1 else 'Forged Copy')
    ax1.axis('off')
    ax2.set_title('Genuine Copy' if label2 == 1 else 'Forged Copy')
    ax2.axis('off')
    plt.show()


# In[24]:


visualize_sample_signature()


# In[25]:


visualize_sample_signature()


# In[30]:


# All the images will be converted to the same size before processing changing the size (different from paper)
img_h, img_w = 155, 220


# In[31]:


visualize_sample_signature()


# In[32]:


visualize_sample_signature()


# In[33]:


visualize_sample_signature()


# In[34]:


visualize_sample_signature()


# In[35]:


visualize_sample_signature()


# In[36]:


visualize_sample_signature()


# In[37]:


visualize_sample_signature()


# In[38]:


def generate_batch(data_groups, batch_size = 32):
    '''Function to generate a batch of data with batch_size number of data points
    Half of the data points will be Genuine-Genuine pairs and half will be Genuine-Forged pairs'''
    while True:
        images = []
        labels = []
        for author in data_groups:
            for image in author:
                images.append(image)
        
        images = shuffle(images)
        k = 0
        batch=np.zeros((batch_size, img_h, img_w, 1))
        targets=np.zeros((batch_size,))
        for ix, image in enumerate(images):
            img1 = cv2.imread(image, 0)
            img1 = cv2.resize(img1, (img_w, img_h))
            img1 = np.array(img1, dtype = np.float64)
            img1 /= 255
            img1 = img1[..., np.newaxis]
            batch[k, :, :, :] = img1
            targets[k] = 0 if 'F' in image else 1
            k += 1
            if k == batch_size:
                yield batch, targets
                k = 0
                batch=np.zeros((batch_size, img_h, img_w, 1))
                targets=np.zeros((batch_size,))


# In[39]:


def euclidean_distance(vects):
    '''Compute Euclidean Distance between two vectors'''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# In[41]:


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# In[42]:


import keras
from keras import models
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer, InputSpec
from keras.regularizers import l2
from keras import backend as K
import keras.backend.tensorflow_backend as tfback
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.multi_gpu_utils import multi_gpu_model


# In[66]:


#Base CNN for our implementation
def CNN_sig(input_shape):
    '''Base Network'''
    
    seq = Sequential()
    seq.add(Conv2D(40, kernel_size=(7, 7), activation='relu', name='conv1_1', strides=1, input_shape= input_shape, 
                        init='glorot_uniform', dim_ordering='tf'))
    seq.add(MaxPooling2D((2,2), strides=(2, 2)))    
    
    seq.add(Conv2D(30, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, init='glorot_uniform',  dim_ordering='tf'))
    seq.add(MaxPooling2D((3,3), strides=(3, 3)))
 
    
    seq.add(Conv2D(20, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, init='glorot_uniform',  dim_ordering='tf'))
    seq.add(MaxPooling2D((3,3), strides=(3, 3)))
    seq.add(Dropout(0.3))# added extra
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(160, activation='softmax'))
    seq.add(Dropout(0.5))
    seq.add(Dense(1))
    
    return seq


# In[67]:


input_shape=(img_h, img_w, 1)


# In[68]:


# network definition
base_network = CNN_sig(input_shape)

input_a = Input(shape=(input_shape))
# input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
# processed_b = base_network(input_b)

# Compute the Euclidean distance between the two vectors in the latent space
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a])
# model = Model(input=[input_a, input_b], output=distance)
model = Model(input=[input_a], output=processed_a)


# In[69]:


import tensorflow as tf


# In[70]:


#optimizer stotestic gradient desent with momentum as per paper
opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)


# In[79]:


rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)


# In[80]:


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# In[81]:


#values of loss, learning rate and optimizer are taken as per paper
model.compile(loss=contrastive_loss, optimizer=rms)


# In[82]:


model.summary()


# In[91]:



# Using Keras Callbacks, save the model after every epoch
# Reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 epochs
# Stop the training using early stopping if the validation loss does not improve for 12 epochs
callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('./BHSig260/Weights/cnn-bhsig260-full-xavier-{epoch:03d}.h5', verbose=1, save_weights_only=True)
]


# In[92]:


batch_sz = 32
num_train_samples = train_size
num_val_samples = val_size
num_test_samples = test_size
num_train_samples, num_val_samples, num_test_samples


# In[93]:


results = model.fit_generator(generate_batch(train, batch_sz),
                              steps_per_epoch = num_train_samples//batch_sz,
                              epochs = 100,
                              validation_data = generate_batch(val, 10),
                              validation_steps = num_val_samples//10,
                              callbacks = callbacks)


# In[96]:


def compute_accuracy_roc(predictions, labels, plot_far_frr =False):
    '''
    Compute ROC accuracy with a range of thresholds on distances.
    Plot FAR-FRR and P-R curves to measure performance on input set
    '''
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1) #similar
    ndiff = np.sum(labels == 0) #different
    step = 0.00001
    max_acc = 0
    best_thresh = -1
    frr_plot = []
    far_plot = []
    pr_plot = []
    re_plot = []
    ds = []
    for d in np.arange(dmin, dmax+step, step):
        idx1 = predictions.ravel() <= d #guessed genuine
        idx2 = predictions.ravel() > d #guessed forged
        tp = float(np.sum(labels[idx1] == 1))
        tn = float(np.sum(labels[idx2] == 0))
        fp = float(np.sum(labels[idx1] == 0))
        fn = float(np.sum(labels[idx2] == 1))
#         print(tp, tn, fp, fn)
        tpr = float(np.sum(labels[idx1] == 1)) / nsame       
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        
        
        acc = 0.5 * (tpr + tnr)
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
#       print ('ROC', acc, tpr, tnr)
       
        if (acc > max_acc):
            max_acc, best_thresh = acc, d
        
        #if (fp+tn) != 0.0 and (fn+tp) != 0.0:
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        frr_plot.append(frr)
        pr_plot.append(pr)
        re_plot.append(re)
        far_plot.append(far)
        ds.append(d)
            
    
    if plot_far_frr:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(121)
        ax.plot(ds, far_plot, color = 'red')
        ax.plot(ds, frr_plot, color = 'blue')
        ax.set_title('Error rate')
        ax.legend(['FAR', 'FRR'])
        ax.set(xlabel = 'Thresholds', ylabel = 'Error rate')
        
        ax1 = fig.add_subplot(122)
        ax1.plot(ds, pr_plot, color = 'green')
        ax1.plot(ds, re_plot, color = 'magenta')
        ax1.set_title('P-R curve')
        ax1.legend(['Precision', 'Recall'])
        ax.set(xlabel = 'Thresholds', ylabel = 'Error rate')
        
        plt.show()
    return max_acc, best_thresh


# In[97]:


# Load the weights from the epoch which gave the best validation accuracy

def load_and_check_model(weight):
    model.load_weights(weight)

    val_gen = generate_batch(val, 1)
    pred, tr_y = [], []
    for i in range(num_val_samples):
        img1, label = next(val_gen)
        tr_y.append(label)
        pred.append(model.predict(img1))

    tr_acc, threshold = compute_accuracy_roc(np.array(pred), np.array(tr_y))
    return tr_acc, threshold


# In[98]:


# BHSIG260 full dataset with Xavier Initialisation on Uniform Distribution
acc_thresh = []
for i in range(1,18,1):
    acc_thresh.append(load_and_check_model('./BHSig260/Weights/cnn-bhsig260-full-xavier-'+str(i).zfill(3)+'.h5'))
    print('For model '+str(i)+' Validation Accuracy = ',acc_thresh[i-1][0]*100,'%')


# In[99]:


def test_model(weight):
    model.load_weights(weight)

    test_gen = generate_batch(test, 1)
    pred, tr_y = [], []
    for i in range(num_test_samples):
        img1, label = next(test_gen)
        tr_y.append(label)
        pred.append(model.predict(img1))

    tr_acc, threshold = compute_accuracy_roc(np.array(pred), np.array(tr_y), plot_far_frr = True)
    return tr_acc, threshold


# In[100]:


acc, threshold = test_model('./BHSig260/Weights/cnn-bhsig260-full-xavier-002.h5')
acc, threshold


# In[101]:


"""Accuracy = 70.00% and Threshold = 0.16 Using Xavier Initialisation on Uniform distribution"""


# In[102]:


def predict_score():
    '''Predict distance score and classify test images as Genuine or Forged'''
    test_gen = generate_batch(test, 1)
    test_point, test_label = next(test_gen)
    img1 = test_point
    print('True Label = ',test_label)
    fig, (ax1) = plt.subplots(1, 1, figsize = (8, 8))
    ax1.imshow(np.squeeze(img1), cmap='gray')
    result = model.predict(img1)
    print("Predicted Label = ", result)
    if result > threshold:
        print("Its a Genuine Signature")
        ax1.set_title('Genuine')
    else:
        print("Its a Forged Signature")
        ax1.set_title('Forged')
    ax1.axis('off')


# In[103]:


predict_score()


# In[104]:


predict_score()


# In[105]:


predict_score()


# In[106]:


predict_score()


# In[ ]:




