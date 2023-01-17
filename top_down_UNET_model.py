import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from pix2pix import Pix2pix
import time
import sys
import os
import csv
#import get_metrics
import pickle


# example of pix2pix gan for satellite to map image-to-image translation
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
from keras.layers import Lambda
import keras



#load and prepare training images
#def load_real_samples(filename):
    # load compressed arrays
    #data = load(filename)
    # unpack arrays
    #X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    #X1 = (X1 - 127.5) / 127.5
    #X2 = (X2 - 127.5) / 127.5
    #return [X1, X2]
 
# select a batch of random samples, returns images and target
def generate_real_samples(A, B, batch_size, patch_shape1, patch_shape2):


    B = (2. * B - 1.)#.reshape(batch_size, im_rows, im_cols, n_output_channels)
    #print("b shape is {}".format(b.shape))

    #????
    #
    y = ones((batch_size, patch_shape1, patch_shape2, 1))
    return [A, B], y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape1, patch_shape2):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape1, patch_shape2, 1))
    return X, y

def train_model(A,B, batch_size, patch_shape1, patch_shape2, d_model, g_model, gan_model):
        #print("top trained model")
        #print(d_model)
        #print(g_model)
        #print(gan_model)
        #def generate_real_samples(A, B, asub, bsub, batch_size, patch_shape):
        [X_realA, X_realB], y_real = generate_real_samples(A, B, batch_size, patch_shape1, patch_shape2)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, patch_shape1, patch_shape2)
        
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        #print("dloss1 {}".format(d_loss1))
        #print("dloss2 {}".format(d_loss2))

        
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        #print("dloss2 {}".format(d_loss2))
        #print("this happened")
        # update the generator
        #print(patch_shape1, patch_shape2)
        #print(X_fakeB.shape)
        #print(y_real.shape)
        #print(X_realB.shape)
        
        
        
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        return d_loss1, d_loss2, g_loss
        # summarize performance

        
        
def define_discriminator(input_image_shape, output_image_shape):
    #initialize the weights
    init = RandomNormal(stddev=0.02)
    # source image input
    
    #3d (input)
    in_src_image = Input(shape=input_image_shape)
    # target image input
    
    #2d (output)
    in_target_image = Input(shape=output_image_shape)
    # concatenate images channel-wise
    #merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_target_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
        #64 by 64
    merged2 = Concatenate()([d, in_src_image])
    
    
    # C256
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged2)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    #8 by 8
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    #4 by 4
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    
    
    
    
    
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model




# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    #if skip_in is not None:
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g
 
def decoder_block_no_skip(layer_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    
    # relu activation
    g = Activation('relu')(g)
    return g
 
 
 
 
# define the standalone generator model
def define_generator(image_shape, n_output_channels):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    #e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e5, 512)
    d2 = decoder_block(d1, e4, 512)
    d3 = decoder_block(d2, e3, 512)
    d4 = decoder_block(d3, e2, 512, dropout=False)
    d5 = decoder_block(d4, e1, 256, dropout=False)
    d6 = decoder_block_no_skip(d5, 128, dropout = False)
    d7 = decoder_block_no_skip(d6, 64, dropout = False)
    #d6 = decoder_block(d5, None, 128, dropout=False)
    #d7 = decoder_block(d6, None, 64, dropout=False)
    # output

    #HERE I BELIEVE THE 3 SHOULD BE CHANGED TO THE NUMBER OF OUTPUT CHANNELS
    g = Conv2DTranspose(n_output_channels, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model




# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,20])
    return model
        
        