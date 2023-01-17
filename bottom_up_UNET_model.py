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




# select a batch of random samples, returns images and target
def generate_real_samples(A, B, batch_size, patch_shape1, patch_shape2):

    #rnd_arr = np.random.permutation(B.shape[0])
    #f32 = tuple(rnd_arr[:batch_size]) 
    #a = A[f32, :, :, :].reshape(batch_size, im_rows, im_cols, A.shape[3])
    
    #a = a1[:,:,:,asub]
    #print("a shape is {}".format(a.shape))
    #b1 = B[f32,:,:,:].reshape(batch_size, b_rows, b_cols, B.shape[3])
    #b = (2. * b1 - 1.)#.reshape(batch_size, im_rows, im_cols, n_output_channels)
    #print("b shape is {}".format(b.shape))

    #????
    #
    y = ones((batch_size, patch_shape1, patch_shape2, 1))
    return [A, B], y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape1, patch_shape2):
    # generate fake instance
    
    #add output here for generator
    X, rot,_,_ = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape1, patch_shape2, 1))
    return X, y


def train_model(A,B, rot_array, pos_array, shape_array, 
        batch_size, patch_shape1, patch_shape2, d_model, g_model, gan_model):
        
        #def generate_real_samples(A, B, asub, bsub, batch_size, patch_shape):
        
        #just A and B
        [X_realA, X_realB], y_real = generate_real_samples(A, B, batch_size, patch_shape1, patch_shape2)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, patch_shape1, patch_shape2)
        
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        
        #rounded fake
        rfake = X_fakeB.copy()
        rfake[(rfake > .6)] = 1
        rfake[(rfake < .4)] = 0
        
        #d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        d_loss2 = d_model.train_on_batch([X_realA, rfake], y_fake)
        
        # update the generator
        rot_1 = rot_array[:,0]
        rot_2 = rot_array[:,1]
        #fake array
        #rot_1 = np.array([0,1,0,0])
        g_loss, pixel_loss, rot_loss_1, rot_loss_2, shape_loss,_ =\
                        gan_model.train_on_batch(X_realA, 
                        [y_real, X_realB, rot_1, rot_2, shape_array])
        # summarize performance

        return d_loss1, d_loss2, g_loss, pixel_loss, rot_loss_1, rot_loss_2, shape_loss

def train_hybrid(A,B, halfB, rot_array, pos_array, shape_array, 
        batch_size, patch_shape1, patch_shape2, td_g_model, d_model, g_model, pc_arch_model):
        
        #def generate_real_samples(A, B, asub, bsub, batch_size, patch_shape):
        #just A and B
        [X_realA, X_realB], y_real = generate_real_samples(A, B, batch_size, patch_shape1, patch_shape2)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, patch_shape1, patch_shape2)
        
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        
        #rounded fake
        rfake = X_fakeB.copy()
        rfake[(rfake > .6)] = 1
        rfake[(rfake < .4)] = 0
        
        #d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        d_loss2 = d_model.train_on_batch([X_realA, rfake], y_fake)
        
        # update the generator
        rot_1 = rot_array[:,0]
        rot_2 = rot_array[:,1]
        #fake array
        #rot_1 = np.array([0,1,0,0])
        
        #[in_src,in_gen], [dis_out, gen_out, rot_out1, rot_out2, shape_out]
        #define_pcarch(td_g_model, g_model, d_model, half_image_shape, output_image_shape, final_gen_activation, 
        #       take_difference = True, learning_rate = 0.0002)
        ones = np.ones(halfB.shape)
        p5s = np.ones(halfB.shape) * .5
        g_loss, pixel_loss, rot_loss_1, rot_loss_2, shape_loss,_ =\
                        pc_arch_model.train_on_batch([halfB, X_fakeB, ones, p5s],
                        [y_real, X_realB, rot_1, rot_2, shape_array])
        # summarize performance

        return d_loss1, d_loss2, g_loss, pixel_loss, rot_loss_1, rot_loss_2, shape_loss
        

def define_discriminator(input_image_shape, output_image_shape, learning_rate = .0002):#.00005
    #initialize the weights
    init = RandomNormal(stddev=0.02)
    # source image input
    
    in_src_image = Input(shape=input_image_shape)
    # target image input
    in_target_image = Input(shape=output_image_shape)
    # concatenate images channel-wise
    
    #CHANGES...COMMENTED OUT MERGED, AND CONCAT, INPUT AT C256
    #merged = Concatenate()([in_src_image, in_target_image])
    # C64
    
    #input changed from merged to in_src_image
    d = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    #64 by 64
    merged2 = Concatenate()([d, in_target_image])
    # C256
    
    #32 by 32
    #changed to merged2
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged2)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    #16 by 16
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    #8 by 8
    #bug found, input changed from merged2 to d (so this was skipping the prevous two layers)
    
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    #4 by 4
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=learning_rate, beta_1=0.5)
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
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g
 
# define the standalone generator model
def define_generator(image_shape, n_output_channels, final_gen_activation):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    
    #256
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    #128
    e2 = define_encoder_block(e1, 128)
    #64
    e3 = define_encoder_block(e2, 256)
    #32
    e4 = define_encoder_block(e3, 512)
    #16
    e5 = define_encoder_block(e4, 512)
    #8
    e6 = define_encoder_block(e5, 512)
    #4
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    
    #2
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    
    #None, 1, 1, 512
    b = Activation('relu')(b)
    bmid = keras.layers.core.Reshape([512])(b)

    #shape_input = b[:,:,:,8:16]
    
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    
    #COMMENTED FOLLOWING BLOCKS AND CHANGED g INPUT
    #d6 = decoder_block(d5, e2, 128, dropout=False)
    #d7 = decoder_block(d6, e1, 64, dropout=False)
    # output

    #HERE I BELIEVE THE 3 SHOULD BE CHANGED TO THE NUMBER OF OUTPUT CHANNELS
    g = Conv2DTranspose(n_output_channels, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
    #out_image = Activation('tanh')(g)
    if final_gen_activation == 'leakyrelu':
        out_image = LeakyReLU(alpha=.1)(g)
    elif final_gen_activation == 'tanh':
        out_image = Activation('tanh')(g)
    elif final_gen_activation == 'sigmoid':
        out_image = Activation('sigmoid')(g)
    else:
        raise Exception("activation not recognized")
        
    #rotation_input_1 = Lambda(lambda x: x[:,0,0,:8])(bmid)
    #rotation_input_2 = Lambda(lambda x: x[:,0,0,8:16])(bmid)
    #shape_input = Lambda(lambda x: x[:,0,0,16:24])(bmid)
    #blist = []
    #for i in range(8):
        #blist.append(Lambda(lambda x: x[:, i])(b))
    
    rotation_output_1 = keras.layers.Dense(units = 8, activation = 'softmax')(bmid)
    #rotation_output_1 = keras.layers.Dense(units = 32, activation = 'softmax')(rotation_output_1)
    rotation_output_2 = keras.layers.Dense(units = 8, activation = 'softmax')(bmid)
    shape_output = keras.layers.Dense(units = 4, activation = 'softmax')(bmid)
    
    # define model
    #model = Model(in_image, [out_image, rotation_input_1, 
    #        rotation_input_2, shape_input])
    model = Model(in_image, [out_image, rotation_output_1, rotation_output_2, shape_output])
    return model




# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape, final_gen_activation, learning_rate = 0.0002):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out, rot_out1, rot_out2, shape_out = g_model(in_src)
    
    #rotation_output_1 = keras.layers.Dense(units = 8, activation = 'softmax')(rot_out1)
    #rotation_output_2 = keras.layers.Dense(units = 8, activation = 'softmax')(rot_out2)
    #shape_output = keras.layers.Dense(units = 4, activation = 'softmax')(shape_out)
    
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out, rot_out1, rot_out2, shape_out])
    #model = Model(in_src, [dis_out, gen_out, rotation_output_1, 
    #    rotation_output_2, shape_output])
    # compile model
    opt = Adam(lr=learning_rate, beta_1=0.5)
    
    if final_gen_activation == 'leakyrelu':
        pixel_loss = 'mae'
    elif final_gen_activation == 'tanh':
        pixel_loss = 'mae'
    elif final_gen_activation == 'sigmoid':
        pixel_loss = 'binary_crossentropy'
    else:
        raise Exception("activation not recognized")
    
    model.compile(loss=['binary_crossentropy', pixel_loss, 
        'binary_crossentropy', 'binary_crossentropy','binary_crossentropy'], 
        optimizer=opt, loss_weights=[1,100,1,1,1])
    return model
    
    
    # define the combined generator and discriminator model, for updating the generator
def define_pcarch(td_g_model, g_model, d_model, half_image_shape, output_image_shape, final_gen_activation, 
                take_difference = True, learning_rate = 0.0002):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    
    #input image wo zeros
    in_src = Input(shape=half_image_shape)
    
    #output predicted by generator with zeros
    in_gen = Input(shape=output_image_shape)
    ones = Input(shape=half_image_shape)
    p5s = Input(shape=half_image_shape)
    #zeros = Input(shape=half_image_shape)
    
    
    # connect the source image to the generator input

    # src image as input, generated image and classification output
    
    td_predict_2d =td_g_model(in_gen)
    
    td_predict_2d = keras.layers.Add()([td_predict_2d, ones])
    td_predict_2d = keras.layers.Multiply()([td_predict_2d, p5s])
    
    #td_predict_2d = (td_predict_2d + 1)/2
    
    td_error_2d = keras.layers.subtract([in_src, td_predict_2d])
    
    in2 = Concatenate(axis = -1)([in_src, td_predict_2d]) 

    gen_out, rot_out1, rot_out2, shape_out = g_model(in2)
    
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in2, gen_out])

    model = Model([in_src,in_gen, ones, p5s], [dis_out, gen_out, rot_out1, rot_out2, shape_out])

    opt = Adam(lr=learning_rate, beta_1=0.5)
    
    if final_gen_activation == 'leakyrelu':
        pixel_loss = 'mae'
    elif final_gen_activation == 'tanh':
        pixel_loss = 'mae'
    elif final_gen_activation == 'sigmoid':
        pixel_loss = 'binary_crossentropy'
    else:
        raise Exception("activation not recognized")
    
    model.compile(loss=['binary_crossentropy', pixel_loss, 
        'binary_crossentropy', 'binary_crossentropy','binary_crossentropy'], 
        optimizer=opt, loss_weights=[1,100,1,1,1])
    return model