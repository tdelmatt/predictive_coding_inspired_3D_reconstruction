import numpy as np
import tensorflow as tf
import time
import sys
import os
import csv
import pickle
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from bottom_up_UNET_model import *
from tc_keras_latest import *

load_model = True
model_load_path = 'models/model_66/best_model'

if load_model:
    d_model = keras.models.load_model(model_load_path + '/discriminator')
    g_model = keras.models.load_model(model_load_path + '/generator')
    print("models successfully loaded from file")

    
A_CV = np.load((cv_path + '/data_2d.npy'))#input cv
B_CV = np.load((cv_path + '/data_3d.npy'))#output cv
    
label_cv = np.load((cv_path + '/data_1d.npy'))
rot_array_cv, pos_array_cv, shape_type_array_cv = prep_1d(label_cv)
    
A_CV = prepare_2d(A_CV)

cv_output, cv_rot_1_out, cv_rot_2_out, cv_shape_out = g_model.predict(A_CV)
cv_output = inverse_transform_3d(cv_output)
cur_l1 = cv_l1_loss = np.sum(np.abs(cv_output - B_CV))


cv_rot_acc_1 = np.sum((np.argmax(cv_rot_1_out, axis = 1)\
        == np.argmax(rot_array_cv[:,0], axis = 1)).astype(int))/rot_array_cv.shape[0]
        
cv_rot_acc_2 = np.sum((np.argmax(cv_rot_2_out,axis = 1)\
        == np.argmax(rot_array_cv[:,1], axis = 1)).astype(int))/rot_array_cv.shape[0]

cv_shape_acc = np.sum((np.argmax(cv_shape_out, axis = 1)\
    == np.argmax(shape_type_array_cv, axis = 1)).astype(int))/shape_type_array_cv.shape[0]

print("corrected cv l1 loss: {}".format(cv_l1_loss))
print("corrected cv rot acc 1: {}".format(cv_rot_acc_1))
print("corrected cv rot acc 2: {}".format(cv_rot_acc_2))


