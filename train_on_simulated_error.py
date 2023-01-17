import numpy as np
import tensorflow as tf
import time
import sys
import os
import csv
import pickle
import subprocess
import imageio
import matplotlib.pyplot as plt

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

#from enlarged_discrim1 import *
from bottom_up_UNET_model import *


def obtain_batch_list():
    start = "shape_data/"
    batch_path_list = []

    for i in range(300):
        temp_batch_dir = start + "batch_{}".format(i)
        if os.path.isdir(temp_batch_dir)\
            and os.path.isfile(temp_batch_dir + "/data_1d.npy")\
            and os.path.isfile(temp_batch_dir + "/data_2d.npy")\
            and os.path.isfile(temp_batch_dir + "/data_3d.npy"):
            batch_path_list.append(temp_batch_dir)
    assert(len(batch_path_list) > 30)
    print("len batch_path_list is {}".format(len(batch_path_list)))
    return batch_path_list


def load_random_batch(batch_path_list, verbose = False):
    batch_ind = np.random.randint(0,len(batch_path_list))
    batch_path = batch_path_list[batch_ind]
    
    if verbose:
        print("batch path is {}".format(batch_path))
    
    data_label = np.load(batch_path + "/data_1d.npy")
    data_2d = np.load(batch_path + "/data_2d.npy")
    data_3d = np.load(batch_path + "/data_3d.npy")
    return data_label, data_2d, data_3d


add_3_channels = True
def prepare_2d(data_2d, div = 255.0, error_in = None):
    n,h,w,c = data_2d.shape
    assert(c == 4)
    data_2d = data_2d[:,:,:,:3]
    print(data_2d.shape)
    n,h,w,c = data_2d.shape
    assert(c == 3)
    data_2d = data_2d / div
    print("max 2d after div {}".format(np.max(data_2d)))
    print("min 2d after div {}".format(np.min(data_2d)))
    
    if add_3_channels:
        empty = np.zeros(data_2d.shape)
        data_2d = np.concatenate([data_2d, empty], axis = -1)
        assert(2*c == data_2d.shape[3])
        
    return data_2d


def inverse_transform_2d(data_2d, div=255):
    if add_3_channels:
        raise Exception("this function wont work while add 3 channels is True")
    data_2d *= 255
    return data2d

#final_gen_activation = 'leakyrelu' #mae #sigmoid
final_gen_activation = 'sigmoid'


#sub changed to 0 for leaky relu    
#change sub back to 0.5 for tanh
def prepare_3d(data_3d, mul = 1.0, sub=0.0):
    if final_gen_activation == 'leakyrelu' or 'sigmoid':
        assert(mul == 1.0 and sub == 0.0)
    elif final_gen_activation == 'mae':
        assert(mul == 1.0 and sub == 0.5)
    else:
        raise Exception("activation not recognized")

    return (mul * data_3d - sub)


def inverse_transform_3d(data_3d, mul = 1.0, sub=0.0):
    if final_gen_activation == 'leakyrelu' or 'sigmoid':
        assert(mul == 1.0 and sub == 0.0)
    elif final_gen_activation == 'mae':
        assert(mul == 1.0 and sub == 0.5)
    else:
        raise Exception("activation not recognized")
    data_3d = (data_3d + sub) / mul
    return data_3d


def convert_oh(x, n_vals):
    new = tf.one_hot(x, depth = n_vals).numpy()
    new_shape = list(x.shape)
    new_shape.append(n_vals)
    assert(tuple(new_shape) == new.shape)
    return new

    
def prep_1d(data_1d):

    rot_array = data_1d[:,2:4]
    rot_array = rot_array / 22.5
    rot_array = convert_oh(rot_array.astype(int), 8)
    pos_array = data_1d[:,1:2]
    shape_type_array = data_1d[:,4]
    shape_type_array = convert_oh(shape_type_array.astype(int), 4)
    return rot_array, pos_array, shape_type_array
    

batch_path_list = obtain_batch_list()
cur_iteration = 0
iter_per_job = 10000

outer_path = "models/"
if not os.path.isdir(outer_path):
    os.mkdir(outer_path)

for i in range(1000):
    temp_name = outer_path + "model_{}".format(i)
    if not os.path.isdir(temp_name):
        os.mkdir(temp_name)
        break

complete_outer_dir = temp_name
model_save_path = complete_outer_dir + '/model.ckpt'
g_d_loss_path = complete_outer_dir + '/g_d_loss.csv'
cur_min_path = complete_outer_dir + '/current_minimum.pk1'
best_model_save_path = complete_outer_dir + '/best_model'
output_csv_path = complete_outer_dir + '/progress.csv'

if not os.path.isdir(best_model_save_path):
    os.mkdir(best_model_save_path)

l1_weight = 100
lr = .0002
batch_size = 1
gan_weight = 1
train_path = "shape_data/batch_64"
cv_path = "shape_data/cross_validation_data/batch_13"

if os.path.isfile(cur_min_path):
    cmin_file = open(cur_min_path, 'rb')
    cur_min = pickle.load(cmin_file)
    print("loaded from file cur_min is {}".format(cur_min))
    cmin_file.close()
else:
    cur_min = None
    print("ERROR CURRENT MIN WAS NOT LOADED, BUT IT SHOULD HAVE BEEN!!!")


if cur_iteration == 0:

    #instantiate csvfile
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect=csv.excel)
        writer.writerow(["batch_size: ", batch_size, "l1 weight:", l1_weight,])
        writer.writerow(['iteration', 'new min reached',  'cv l1',"cv_rot_acc_1", "cv_rot_acc_2", "cv_shape_acc", 'cv min', 'cv max','train l1', 'train min', 'train max'])

    with open(g_d_loss_path, 'a', newline='') as csvfile1:
        writer1 = csv.writer(csvfile1, dialect=csv.excel)
        writer1.writerow(['step', 'gloss_curr', 'dloss_curr'])


start_time = time.time()
print("REMEMBER TO MAKE THE DIRECTORY IN MODEL TRIALS, IMAGES, AND MODELS!!!")
iters = iter_per_job # taken from pix2pix paper ยง5.2
start_iter = cur_iteration
end_iter = cur_iteration + iters
memmap = False


if memmap is True:
    
    A = np.load((train_path + '/Xdata.npy'), mmap_mode = 'r')#input train
    B = np.load((train_path + '/Ydata.npy'), mmap_mode = 'r')#output train
    A_CV = np.load((cv_path + '/Xdata.npy'))#input cv
    B_CV = np.load((cv_path + '/Ydata.npy'))#output cv

    #calculate subsample array
    if channel_subsample_x is not None:
        asub = channel_subsample_x
    else:
        asub = list(range(A.shape[3]))

    if channel_subsample_y is not None:
        bsub = channel_subsample_y
    else:      
        bsub = list(range(B.shape[3]))

    print("asub is {}".format(asub))
    print("bsub is {}".format(bsub))

else:

    A = np.load((train_path + '/data_2d.npy'))#input train
    B = np.load((train_path + '/data_3d.npy'))#output train

    A_CV = np.load((cv_path + '/data_2d.npy'))#input cv
    B_CV = np.load((cv_path + '/data_3d.npy'))#output cv
    
    label_cv = np.load((cv_path + '/data_1d.npy'))
    rot_array_cv, pos_array_cv, shape_type_array_cv = prep_1d(label_cv)


A_CV = prepare_2d(A_CV)
print("cv end sum")
print(np.sum(A_CV[:,:,:,3:]))
A_CV_entire = A_CV.copy()
B_CV_entire = B_CV.copy()

n_input_channels = A_CV.shape[3]
n_output_channels = B_CV.shape[3]
im_cols = A.shape[2]
b_rows = B.shape[1]
b_cols = B.shape[2]

im_dim_tup = (im_rows, im_cols)
gradient_issue_count = 0

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


output_image_shape = B_CV.shape[1:]
input_image_shape = A_CV.shape[1:]
load_model = False
model_load_path = 'models/model_113/best_model'

if load_model:
    
    d_model = keras.models.load_model(model_load_path + '/discriminator')
    g_model = keras.models.load_model(model_load_path + '/generator')
    print("models successfully loaded from file")
    cur_min = None

else:

    # define the models
    d_model = define_discriminator(input_image_shape, output_image_shape)
    print("\nDiscriminator Summary")
    print(d_model.summary())

    g_model = define_generator(input_image_shape, n_output_channels, final_gen_activation)
    print("\nGenerator Summary")
    print(g_model.summary())


# define the composite model
gan_model = define_gan(g_model, d_model, input_image_shape, final_gen_activation)
# train model
print("\nGAN model Summary")
print(gan_model.summary())

# determine the output square shape of the discriminator
patch_shape1 = d_model.output_shape[1]
patch_shape2 = d_model.output_shape[2]

train_on_error = True
if train_on_error:
    train_output_dir = complete_outer_dir + '/errors'
    if not os.path.isdir(train_output_dir):
        os.mkdir(train_output_dir)


def obtain_error_path(base_path):

    for i in range(100000):
        temp_name = base_path + "/error_{}".format(i)
        if not os.path.isdir(temp_name):
            os.mkdir(temp_name)
            return temp_name


def read_crop_im(impath, change = False):

    im = imageio.imread(impath)
    im_crop = im[54:310,45:301].copy()
    
    im_crop = im_crop[:,:,:3]
    m,n,c = im_crop.shape
    assert(c == 3)
    assert(m==n and n==256)
    if change:
        os.remove(impath)
        plt.imsave(impath, im_crop)
    return im_crop


def combine_im(data_2d, predict):

    if len(data_2d.shape) == 4:
        
        #data_2d = data_2d.reshape(data_2d.shape[1:])
        real_data = data_2d[:,:,:,:3]
        zeros = data_2d[:,:,:,3:]
    elif len(data_2d.shape) == 3:
        real_data = data_2d[:,:,:3]
        zeros = data_2d[:,:,3:]
    else:
        raise Exception("improper shape")
    print(np.sum(zeros))
    assert(np.sum(zeros) == 0)
    assert(np.sum(real_data > 0))
    predict = predict.reshape([1] + list(predict.shape))
    
    assert(real_data.shape == predict.shape)
    error = real_data - predict
    
    ret_data = np.concatenate([real_data, error], axis = -1)
    assert(6 == data_2d.shape[-1])
    return ret_data


def get_error_data(g_model, data_2d, batch_size, train_output_dir):

    error_path = obtain_error_path(train_output_dir)

    #get prediction
    prediction, _,_,_ = g_model.predict(data_2d)
    print(prediction.shape)
    prediction = prediction.reshape(64,64,64)
    np.save(error_path + '/output.npy', prediction)

    #plot prediction
    path_to_exe = "C:/Users/tdelmatt/Anaconda3/envs/plotlyenv/python"
    to_run = "C:/Users/tdelmatt/Anaconda3/envs/plotlyenv/python plot_3d.py " + error_path
    
    result = subprocess.run(to_run, stdout=subprocess.PIPE)
    print(result.stdout)
    
    impath = error_path + '/3d_predict.png'
    if os.path.isfile(impath):
        print("file exists!!!")
        im = read_crop_im(impath, change = True)
        im = im/255.
        assert(np.sum(im) > 1000 and np.max(im) <= 1)
        A_combined = combine_im(data_2d, im)
        return A_combined
    else:
        print("file does not exist")
        return None


whole_model_plot = 'entire_model.png'
step = -1
for batch in range(start_iter, end_iter):
    
    #MAYBE ADD ANOTHER LOOP HERE, AND SELECT A,B HERE
    #load, prepare train data (taken after for loop for recording metrics)
    data_label, data_2d, data_3d = load_random_batch(batch_path_list)
    data_2d = prepare_2d(data_2d)
    data_3d = prepare_3d(data_3d)
    rot_array, pos_array, shape_type_array = prep_1d(data_label)
    
    n_iter = int(data_2d.shape[0] / batch_size)
    print("n_iter is {}".format(n_iter))
    
    index = 0
    for i in range(n_iter):
    
        #step += 1
        A = data_2d[index:index + batch_size]
        B = data_3d[index:index + batch_size]
        rot = rot_array[index:index + batch_size]
        pos = pos_array[index:index + batch_size]
        shape = shape_type_array[index:index + batch_size]
        
        index += batch_size
        print("index is {}".format(index))

        if train_on_error:
            A = get_error_data(g_model, A, batch_size, train_output_dir)
            if A is None:
                continue
            
        step += 1
        d_loss1, d_loss2, g_loss, pixel_loss, rot_loss_1, rot_loss_2, shape_loss = train_model(A,B, rot, pos, shape,
                batch_size, patch_shape1, patch_shape2, d_model, g_model, gan_model)
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (step+1, d_loss1, d_loss2, g_loss))
        print("pixel loss: [%.3f] rot loss 1: [%.3f] rot loss 2: [%.3f] shape loss: [%.3f]" % (pixel_loss, 
                rot_loss_1, rot_loss_2, shape_loss))


    
        if step == 0:
            asavedir = complete_outer_dir + '/a'
            bsavedir = complete_outer_dir + '/b'
        
        if np.isnan(g_loss) or np.isnan(d_loss1) or np.isnan(d_loss2):
            print("GRADIENT ISSUE AT iteration {}".format(step))

            gradient_issue_count += 1
            if gradient_issue_count > 30:
                print("PYTHON SCRIPT CANCELED DUE TO 30 GRADIENT ISSUES")
                quit()

            if step < 2000:
                #Todo: REINITIALIZE VARIABLES OF MODEL
                #sess.run(tf.global_variables_initializer())
                pass
            else:
                #Todo: RESTORE LAST SAVED VERSION OF MODEL
                #saver.restore(sess, model_save_path)
                pass
            continue

        if step % 20 == 0:

            rnd_arr = np.random.permutation(B.shape[0])
            
            #assert that a_CV has been preprocessed
            if add_3_channels:
                assert(A_CV.shape[3] == 6)
            else:
                assert(A_CV.shape[3] == 3)
            
            #assert B_CV has not been preprocessed
            assert(np.max(B_CV) == 1)
            
            not_gen = True
            while not_gen:
            #rot_array_cv, pos_array_cv, shape_type_array_cv 
                batch = A_CV_entire.shape[0]
                rand_b = np.random.randint(0,batch)
                A_CV = A_CV_entire[rand_b:rand_b+1]#.reshape([1] + list(A_CV_entire.shape[1:]))
                B_CV = B_CV_entire[rand_b:rand_b+1]#.reshape([1] + list(B_CV_entire.shape[1:]))
                
                print("cv end sum in run")
                print(np.sum(A_CV[:,:,:,3:]))
                
                print(A_CV.shape)
                A_CV = get_error_data(g_model, A_CV, batch_size, train_output_dir)
                if A is None:
                    continue
                else:
                    not_gen = False

            cv_output, cv_rot_1_out, cv_rot_2_out, cv_shape_out = g_model.predict(A_CV)
            cv_output = inverse_transform_3d(cv_output)
            #cv loss
            cur_l1 = cv_l1_loss = np.sum(np.abs(cv_output - B_CV)) * 16

            #cv rot accuracy
            cv_rot_acc_1 = np.sum(cv_rot_1_out * rot_array_cv[rand_b,0])#/rot_array_cv.shape[0]
            cv_rot_acc_2 = np.sum(cv_rot_2_out * rot_array_cv[rand_b,1])#/rot_array_cv.shape[0]
            
            cv_shape_acc = np.sum(cv_shape_out * shape_type_array_cv[rand_b])#/shape_type_array_cv.shape[0]

            train_output, tr_rot_1_out, tr_rot_2_out, tr_shape_out = g_model.predict(data_2d)
            train_output = inverse_transform_3d(train_output)
            train_target = inverse_transform_3d(data_3d)

            #cv loss
            train_l1_loss = np.sum(np.abs(train_output\
                                - train_target))
            
            print("train l1 loss: {}".format(train_l1_loss))
            print("cv l1 loss: {}".format(cv_l1_loss))
            print("cv rot acc 1: {}".format(cv_rot_acc_1))
            print("cv rot acc 2: {}".format(cv_rot_acc_2))
            print("cv shape acc{}".format(cv_shape_acc))
            
            cvmax = np.max(cv_output)
            cvmin = np.min(cv_output)
            print("cv max is {}".format(cvmax))
            print("cv min is {}".format(cvmin))
            trmax = np.max(B_CV)
            trmin = np.min(B_CV)
            print("cv target max is {}".format(trmax))
            print("cv target min is {}".format(trmin))
            
            cv_rot_acc_1 = np.sum((np.argmax(cv_rot_1_out, axis = 1)\
                == np.argmax(rot_array_cv[rand_b,0].reshape(1,8), axis = 1)).astype(int))#/rot_array_cv.shape[0]
        
            cv_rot_acc_2 = np.sum((np.argmax(cv_rot_2_out,axis = 1)\
                == np.argmax(rot_array_cv[rand_b,1].reshape(1,8), axis = 1)).astype(int))#/rot_array_cv.shape[0]

            cv_shape_acc = np.sum((np.argmax(cv_shape_out, axis = 1)\
                == np.argmax(shape_type_array_cv[rand_b].reshape(1,4), axis = 1)).astype(int))#/shape_type_array_cv.shape[0]

            print("\n\ncorrected cv shape acc: {}".format(cv_shape_acc))
            print("corrected cv rot acc 1: {}".format(cv_rot_acc_1))
            print("corrected cv rot acc 2: {}".format(cv_rot_acc_2))
            new_min_reached = False

            if cur_min is None or cur_l1 < cur_min:

                new_min_reached = True
                cur_min = cur_l1
                print("\nnew minimum found.  minimum is {}".format(cur_min))
                print()

                if step > 20:

                    save_string = "/cv_generated_iter_{}".format(step)
                    np.save((complete_outer_dir + save_string), cv_output)

                    #save discriminator
                    d_model.save(best_model_save_path + '\discriminator')
                    #save generator
                    g_model.save(best_model_save_path + '\generator')
                    
                    print("Best Model saved in file: %s" % best_model_save_path)
            
            with open(output_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, dialect=csv.excel)
                writer.writerow([step, str(new_min_reached), cur_l1, cv_rot_acc_1, cv_rot_acc_2, cv_shape_acc, cvmin, cvmax, train_l1_loss, trmin, trmax])
            
            with open(g_d_loss_path, 'a', newline='') as csvfile1:
                writer1 = csv.writer(csvfile1, dialect=csv.excel)
                writer1.writerow([step, g_loss, d_loss1, d_loss2])

                    
        if step % 500 == 0 or step == (end_iter - 1):

            ctime = time.time()
            print("time at iteration {} is {}".format(step, (ctime - start_time)))

#save curmin to file
#curmin should be saved in outer directory
output = open(cur_min_path, 'wb')
pickle.dump(cur_min, output,2)
output.close()

