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

#from enlarged_discrim1 import *

from bottom_up_UNET_model import *
import top_down_UNET_model as td


def obtain_batch_list():
    start = "shape_data/"
    batch_path_list = []

    for i in range(10,500):
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
    #print(data_2d.shape)
    n,h,w,c = data_2d.shape
    assert(c == 3)
    data_2d = data_2d / div
    #print("max 2d after div {}".format(np.max(data_2d)))
    #print("min 2d after div {}".format(np.min(data_2d)))
    
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

    #new = np.zeros(new_shape)
    #new[]
    
def prep_1d(data_1d):
    #batch_size, 11
    rot_array = data_1d[:,2:4]
    #print("rotation array {}".format(rot_array))
    rot_array = rot_array / 22.5
    rot_array = convert_oh(rot_array.astype(int), 8)
    #print("one hot rotation array {}".format(rot_array))
    pos_array = data_1d[:,1:2]

    shape_type_array = data_1d[:,4]
    #print("shape type array {}".format(shape_type_array))
    shape_type_array = convert_oh(shape_type_array.astype(int), 4)
    #print("one hot shape type array {}".format(shape_type_array))
    return rot_array, pos_array, shape_type_array
    
    
    
    
    
batch_path_list = obtain_batch_list()

    
    
#sys.path.append('/data/users/tdelmatto/wrangling_data/wrangling_scripts/build_data/END_TO_END_EXAMPLES/end_to_end_preprocessing_examples')


cur_iteration = 0

#iter_per_job = int(sys.argv[2])
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

#complete_inner_dir = complete_outer_dir + '/images'

#hp_config_str = sys.argv[5]

model_save_path = complete_outer_dir + '/model.ckpt'

#image_save_path = complete_inner_dir + '/images'

g_d_loss_path = complete_outer_dir + '/g_d_loss.csv'

#generated_image_save_path = (image_save_path + '/generated') 

cur_min_path = complete_outer_dir + '/current_minimum.pk1'

best_model_save_path = complete_outer_dir + '/best_model'

output_csv_path = complete_outer_dir + '/progress.csv'



if not os.path.isdir(best_model_save_path):
    os.mkdir(best_model_save_path)

#best_model_save_path = best_model_save_path #+ '/model.ckpt'



#make image save path
#if cur_iteration == 0:
#os.mkdir(image_save_path)
#os.mkdir((image_save_path + '/heatmaps'))
#os.mkdir(generated_image_save_path)

#writer.writerow(['l1_weight', 'lr', 'total_l1_loss', 'mape', 'average_daily_loss'])

#I DONT THINK THESE HYPERPARAMETERS ARE PASSED INTO THE CODE CURRENTLY
l1_weight = 100
lr = .0002
batch_size = 4
gan_weight = 1

#get data paths
train_path = "shape_data/batch_64"
cv_path = "shape_data/cross_validation_data/batch_13"


#if 'min_start' in hp_config_dict:
#    cur_min = hp_config_dict['min_start']

#elif cur_iteration > 0:

if os.path.isfile(cur_min_path):
    cmin_file = open(cur_min_path, 'rb')
    cur_min = pickle.load(cmin_file)
    print("loaded from file cur_min is {}".format(cur_min))
    cmin_file.close()
else:
    cur_min = None
    print("ERROR CURRENT MIN WAS NOT LOADED, BUT IT SHOULD HAVE BEEN!!!")
    #load cur min from file
    #cur min should be saved in outer directory
    #curmin = cur min
    #pass    

#else:
#    cur_min = None


if cur_iteration == 0:

    #instantiate csvfile
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect=csv.excel)
        
                        
        #writer.writerow(row)
        writer.writerow(["batch_size: ", batch_size, "l1 weight:", l1_weight,])
        writer.writerow(['iteration', 'new min reached',  'cv l1',"cv_rot_acc_1", "cv_rot_acc_2", "cv_shape_acc", 'cv min', 'cv max','train l1', 'train min', 'train max'])

    
    with open(g_d_loss_path, 'a', newline='') as csvfile1:
        writer1 = csv.writer(csvfile1, dialect=csv.excel)
        writer1.writerow(['step', 'gloss_curr', 'dloss_curr'])


    


start_time = time.time()

print("REMEMBER TO MAKE THE DIRECTORY IN MODEL TRIALS!!!, AND SUBDIRECTORIES IMAGES, AND MODELS")


#useful parameters would be 
#iterations and model output file directory/name


iters = iter_per_job # taken from pix2pix paper ยง5.2
start_iter = cur_iteration
end_iter = cur_iteration + iters

#ORIGINAL BATCH SIZE
#batch_size = 1 # taken from pix2pix paper ยง5.2

memmap = False
#another way to code this would be to try and load in the next few batches by randomly sampling from the
#array ahead of time and potentially doing this loading in parallel
if memmap is True:
    
    A = np.load((train_path + '/Xdata.npy'), mmap_mode = 'r')#input train
    B = np.load((train_path + '/Ydata.npy'), mmap_mode = 'r')#output train

    #With 3 pre
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
    

_, td_h, td_w,_  = A_CV.shape
td_output_shape = (td_h, td_w, 3)

A_CV = prepare_2d(A_CV)
#B_CV = prepare_3d(B_CV)
#ADD CHANNEL SUBSAMPLE BLOCK HERE...
#FOR NOW WE ONLY SUBSAMPLE THE INPUT
#if channel_subsample is not None:
"""
if channel_subsample_x is not None:

    #A = A[:,:,:, channel_subsample_x]
    #B = B[:,:,:, channel_subsample]
    A_CV = A_CV[:,:,:, channel_subsample_x]   
    #B_CV = B_CV[:,:,:, channel_subsample]

if channel_subsample_y is not None:
    #B = B[:,:,:, channel_subsample_y]
    B_CV = B_CV[:,:,:, channel_subsample_y]

    #A = A[...channel_subsample...]
"""
    
    
#OBTAIN N_CHANNELS
#THESE ASSERTIONS NO LONGER MAKE SENSE 
#BECAUSE TRAIN AND CROSS VALIDATION SETS ARE 
#REFEERENCED DIFFERENTLY
n_input_channels = A_CV.shape[3]
#assert(n_input_channels == A_CV.shape[3])

n_output_channels = B_CV.shape[3]
#assert(n_output_channels == B_CV.shape[3])

#n_channels_in_plot = (2 * n_output_channels) + n_input_channels

#OBTAIN IMAGE DIMENSIONS
#ASSERT IMAGE DIMENSIONS ARE EVEN MULTIPLE OF 256
#assert(A.shape[1] == B.shape[1] == A_CV.shape[1] == B_CV.shape[1])
#assert(A.shape[2] == B.shape[2] == A_CV.shape[2] == B_CV.shape[2])
im_rows = A.shape[1]
im_cols = A.shape[2]

b_rows = B.shape[1]
b_cols = B.shape[2]

im_dim_tup = (im_rows, im_cols)

gradient_issue_count = 0





 


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
 
"""
#I DONT THINK THIS IS CALLED CURRENTLY
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    patch_shape1 = d_model.output_shape[1]
    patch_shape2 = d_model.output_shape[2]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):


        #note that X_realB or X_fakeB are actually the outputs of the generator y or y'
        #the things we are trying to create using this network.  In the case of the generator,
        #they are inputs concatenated with the generator input X (here named X_realA)
        #that the validity of the output is conditioned on.  The output is a patch of ones or zeros in the
        #discriminator.  So the output/input of the discriminator is different than that of the generator.  

        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, patch_shape1, patch_shape2)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, patch_shape1, patch_shape2)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        #if (i+1) % (bat_per_epo * 10) == 0:
            #summarize_performance(i, g_model, dataset)
 
"""
 
# load image data
#dataset = load_real_samples('maps_256.npz')
#print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
#image_shape = dataset[0].shape[1:]

output_image_shape = B_CV.shape[1:]
input_image_shape = A_CV.shape[1:]



    
    #d_model = define_discriminator(input_image_shape, output_image_shape)
    #print("\nDiscriminator Summary")
    #print(d_model.summary())

    #g_model = define_generator(input_image_shape, n_output_channels, final_gen_activation)
    #print("\nGenerator Summary")
    #print(g_model.summary())
    
load_model = True
model_load_path = 'models/bu_more_data/best_model'

if load_model:
    
    d_model = keras.models.load_model(model_load_path + '/discriminator')
    d_model.name = "dmodel_111"
    g_model = keras.models.load_model(model_load_path + '/generator')
    g_model.name = "gmodel_111"
    print("bottom up models successfully loaded from file")

else:
    # define the models
    d_model = define_discriminator(input_image_shape, output_image_shape)
    print("\nDiscriminator Summary")
    print(d_model.summary())

    g_model = define_generator(input_image_shape, n_output_channels, final_gen_activation)
    print("\nGenerator Summary")
    print(g_model.summary())

    
load_td_model = True
td_model_load_path = 'models/model_174/best_model'

if load_td_model:
    #td_d_model = keras.models.load_model(td_model_load_path + '/tddiscriminator')
    td_g_model = keras.models.load_model(td_model_load_path + '/generator')
    td_g_model.name = "td_g_model111"
    print("top down models successfully loaded from file")
else:
# define the models
    #td_d_model = td.define_discriminator(output_image_shape, td_output_shape)
    #print("\nDiscriminator Summary")
    #print(td_d_model.summary())

    td_g_model = td.define_generator(output_image_shape, td_output_shape[2])
    print("\nGenerator Summary")
    print(td_g_model.summary())
# define the composite model
#td_gan_model = td.define_gan(td_g_model, td_d_model, output_image_shape)

#gan_model = define_gan(g_model, td_d_model, input_image_shape, final_gen_activation)
pc_arch_model = define_pcarch(td_g_model, g_model, d_model, td_output_shape, output_image_shape, final_gen_activation, 
        take_difference = True)

# train model
#print("\npcarch model Summary")
print(pc_arch_model.summary())
#print(td_gan_model.summary())


#print("\nGAN model Summary")
#print(td_gan_model.summary())

# train model
#print("\nGAN model Summary")
#print(gan_model.summary())

#train(d_model, g_model, gan_model, dataset)

# determine the output square shape of the discriminator
#td_patch_shape1 = td_d_model.output_shape[1]
#td_patch_shape2 = td_d_model.output_shape[2]

patch_shape1 = d_model.output_shape[1]
patch_shape2 = d_model.output_shape[2]

# unpack dataset
#trainA, trainB = dataset
# calculate the number of batches per training epoch
#bat_per_epo = int(len(trainA) / n_batch)
# calculate the number of training iterations
#n_steps = bat_per_epo * n_epochs
# manually enumerate epochs

data_label, data_2d, data_3d = load_random_batch(batch_path_list)
data_2d = prepare_2d(data_2d)
data_3d = prepare_3d(data_3d)
rot_array, pos_array, shape_type_array = prep_1d(data_label)


whole_model_plot = 'entire_model.png'

step = -1
for batch in range(start_iter, end_iter):


    #note that X_realB or X_fakeB are actually the outputs of the generator y or y'
    #the things we are trying to create using this network.  In the case of the generator,
    #they are inputs concatenated with the generator input X (here named X_realA)
    #that the validity of the output is conditioned on.  The output is a patch of ones or zeros in the
    #discriminator.  So the output/input of the discriminator is different than that of the generator.  

    # select a batch of real samples
    
    #MAYBE ADD ANOTHER LOOP HERE, AND SELECT A,B HERE

    
    n_iter = int(data_2d.shape[0] / batch_size)
    print("n_iter is {}".format(n_iter))
    
    index = 0
    for i in range(n_iter):
    
        #load, prepare train data (taken after for loop for recording metrics)
        data_label, data_2d, data_3d = load_random_batch(batch_path_list)
        data_2d = prepare_2d(data_2d)
        data_3d = prepare_3d(data_3d)
        rot_array, pos_array, shape_type_array = prep_1d(data_label)
        
        
        step += 1
        A = data_2d[index:index + batch_size]
        B = data_3d[index:index + batch_size]
        rot = rot_array[index:index + batch_size]
        pos = pos_array[index:index + batch_size]
        shape = shape_type_array[index:index + batch_size]
        
        index += batch_size
        print("index is {}".format(index))
        
        #get prediction
        #plot
        #check if exists
        #if it exists:
            #slice A in half
            #assert first half is not 0, second half is
            #subtract with a
            #combine with A
            #train
            #step += 1
        #else:
            #continue
        
        d_loss1, d_loss2, g_loss, pixel_loss, rot_loss_1, rot_loss_2, shape_loss =\
                train_hybrid(A, B, A[:,:,:,:3], rot, pos, 
                        shape, batch_size, patch_shape1, 
                        patch_shape2, td_g_model, d_model, g_model, pc_arch_model)
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (step+1, d_loss1, d_loss2, g_loss))
        print("pixel loss: [%.3f] rot loss 1: [%.3f] rot loss 2: [%.3f] shape loss: [%.3f]" % (pixel_loss, 
                rot_loss_1, rot_loss_2, shape_loss))


        if step == 0:
            
            asavedir = complete_outer_dir + '/a'
            bsavedir = complete_outer_dir + '/b'
            
            #tf.keras.utils.plot_model(gan_model, to_file = whole_model_plot, show_shapes = True)
            #tf.keras.utils.plot_model(g_model, to_file = 'generator.png', show_shapes = True)
            #tf.keras.utils.plot_model(d_model, to_file = 'discriminator.png', show_shapes = True)

            #if not os.path.isdir(asavedir):
                #os.mkdir(asavedir)
            #if not os.path.isdir(bsavedir):
                #os.mkdir(bsavedir)

            #get_info_dicts.save_images(a, input_chan_dict, date_loc_dict, f32[0], 'in', save_dir = asavedir)
            #get_info_dicts.save_images(b, output_chan_dict, date_loc_dict, f32[0], 'out', save_dir = bsavedir)          
             


        
        if np.isnan(g_loss) or np.isnan(d_loss1) or np.isnan(d_loss2):

            
            #writes an error message to notify that a gradient issue has been encountered
            #with open(output_csv_path, 'a', newline='') as csvfile1:
            #    writer1 = csv.writer(csvfile1, dialect=csv.excel)
            #    to_write = "GRADIENT ISSUE AT iteration {}".format(step)
            #    writer1.writerow([to_write])           
            print("GRADIENT ISSUE AT ITERATION {}".format(step))

                
            #ELIMINATE FOLLOWING ASSERT, AFTER IMPLEMENTATION COMPLETED (FOR NOW THIS ENDS SCRIPT HERE FOLLOWING GRADIENT ISSUE)
            #assert(1 != 1)        

            gradient_issue_count += 1
            if gradient_issue_count > 30:
                print("PYTHON SCRIPT CANCELED DUE TO 30 GRADIENT ISSUES")
                quit()

            if step < 2000:
                #TO COMPLETE: REINITIALIZE VARIABLES OF MODEL
                #sess.run(tf.global_variables_initializer())
                pass
            else:
                #TO COMPLETE: RESTORE LAST SAVED VERSION OF MODEL
                #saver.restore(sess, model_save_path)
                pass
            continue
        
        #write g d loss to csv file for more interactive loss examination
        #with open(g_d_loss_path, 'a', newline='') as csvfile:
        #    writer = csv.writer(csvfile, dialect=csv.excel)
        #    writer.writerow([step, d_loss1, d_loss2, g_loss])
        
        #print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))


        if step % 20 == 0:

            rnd_arr = np.random.permutation(B.shape[0])
            #print("rnd shape is {}".format(rnd_arr.shape))
            #Ars = A[rnd_arr[:30],:,:,:].reshape(30,im_rows, im_cols, A.shape[3])
            #Brs = B[rnd_arr[:30],:,:,:].reshape(30,im_rows, im_cols, B.shape[3])


            #CHANGE
            #output = ((model.sample_generator(sess, Ars[:,:,:,asub], is_training=False) + 1.) / 2.)
            #output = ((g_model.predict(Ars[:,:,:,asub]) + 1.) / 2.)

            #print("train output max is {}".format(np.max(output)))
            #print("train output min is {}".format(np.min(output)))
            
            #print("train target max is {}".format(np.max(Ars[:,:,:,asub])))
            #print("train target min is {}".format(np.min(Ars[:,:,:,asub])))
            
            """
            #assert that a_CV has been preprocessed
            if add_3_channels:
                assert(A_CV.shape[3] == 6)
            else:
                assert(A_CV.shape[3] == 3)
            
            #assert B_CV has not been preprocessed
            assert(np.max(B_CV) == 1)
            
            #rot_array_cv, pos_array_cv, shape_type_array_cv 
            
            td_predict = (td_g_model.predict(B_CV)+ 1) / 2
            td_cv_l1 = cur_l1 = np.sum(np.abs(A_CV[:,:,:,:3] - td_predict))
            
            td_predict_train = (td_g_model.predict(data_3d)+ 1) / 2
            td_tr_l1 = np.sum(np.abs(td_predict_train - data_2d[:,:,:,:3]))
            
            print("top down train l1 loss: {}".format(td_tr_l1))
            print("top down cv l1 loss: {}".format(td_cv_l1))
            
            tdcvmax = np.max(td_cv_l1)
            tdcvmin = np.min(td_cv_l1)
            print("cv max is {}".format(tdcvmax))
            print("cv min is {}".format(tdcvmin))
            tdtrmax = np.max(A_CV)
            tdtrmin = np.min(A_CV)
            print("cv target max is {}".format(tdtrmax))
            print("cv target min is {}".format(tdtrmin))
            """
            
            
            #CHANGE
            #cv_output = ((model.sample_generator(sess, A_CV, is_training=False) + 1.) / 2.)
            #cv_output, cv_rot_1_out, cv_rot_2_out, cv_shape_out = g_model.predict(A_CV)
            gpred, _,_,_ = g_model.predict(A_CV)
            
            dis_out, cv_output, cv_rot_1_out, cv_rot_2_out, cv_shape_out =\
                                pc_arch_model.predict([A_CV[:,:,:,:3],
                                        gpred,np.ones((16,256,256,3)), .5*np.ones((16,256,256,3))])
            cv_output = inverse_transform_3d(cv_output)
            #cv loss
            cur_l1 = cv_l1_loss = np.sum(np.abs(cv_output - B_CV))
            
            #print(cv_rot_1_out)
            #print(rot_array_cv)
            #print(cv_rot_1_out * rot_array_cv[:,0])
            #cv rot accuracy
            cv_rot_acc_1 = np.sum(cv_rot_1_out * rot_array_cv[:,0])/rot_array_cv.shape[0]
            cv_rot_acc_2 = np.sum(cv_rot_2_out * rot_array_cv[:,1])/rot_array_cv.shape[0]
            
            cv_shape_acc = np.sum(cv_shape_out * shape_type_array_cv)/shape_type_array_cv.shape[0]

            """"
            #train_output, tr_rot_1_out, tr_rot_2_out, tr_shape_out = g_model.predict(data_2d)
            dis_out, train_output, tr_rot_1_out, tr_rot_2_out, tr_shape_out = pc_arch_model.predict(data_2d)
            train_output = inverse_transform_3d(train_output)
            train_target = inverse_transform_3d(data_3d)

            #cv loss
            train_l1_loss = np.sum(np.abs(train_output\
                                - train_target))
            
            print("train l1 loss: {}".format(train_l1_loss))
            """
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
                == np.argmax(rot_array_cv[:,0], axis = 1)).astype(int))/rot_array_cv.shape[0]
        
            cv_rot_acc_2 = np.sum((np.argmax(cv_rot_2_out,axis = 1)\
                == np.argmax(rot_array_cv[:,1], axis = 1)).astype(int))/rot_array_cv.shape[0]

            cv_shape_acc = np.sum((np.argmax(cv_shape_out, axis = 1)\
                == np.argmax(shape_type_array_cv, axis = 1)).astype(int))/shape_type_array_cv.shape[0]

            print("\n\ncorrected cv shape acc: {}".format(cv_shape_acc))
            print("corrected cv rot acc 1: {}".format(cv_rot_acc_1))
            print("corrected cv rot acc 2: {}".format(cv_rot_acc_2))
            
            
            #test_out = ((model.sample_generator(sess, np.expand_dims(A[23], axis=0), is_training=False)[0] + 1.) / 2.) 
            #plt.imsave((image_save_path + '/sanitytest23.png'), test_out[:,:,0])

            new_min_reached = False
            if cur_min is None or cur_l1 < cur_min:
                new_min_reached = True
                cur_min = cur_l1
                print("\nnew minimum found.  minimum is {}".format(cur_min))
                print()
                if step > 20:

                    #cv_entire = ((model.sample_generator(sess, A_CV, is_training=False) + 1.) / 2.)
                    save_string = "/cv_generated_iter_{}".format(step)
                    
                    #uncomment to save
                    #np.save((complete_outer_dir + save_string), cv_output)
                    np.save((complete_outer_dir + save_string), cv_output)
                    
                    
                    #save discriminator
                    td_g_model.save(best_model_save_path + '\gtdgenerator')
                    d_model.save(best_model_save_path + '\discriminator')
                    #save generator
                    g_model.save(best_model_save_path + '\generator')
                    
                    print("Best Model saved in file: %s" % best_model_save_path)
                    #TO ADD: SAVE MODEL....
                #instantiate csvfile
            
            with open(output_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, dialect=csv.excel)
                
                #writer.writerow(['iteration', 'new min reached',  'cv l1', 'cv min', 'cv max','train l1', 'train min', 'train max'])    
                
                #previous
                writer.writerow([step, str(new_min_reached), cur_l1, cv_rot_acc_1, cv_rot_acc_2, cv_shape_acc, cvmin, cvmax, None, None, None])
                #writer.writerow([step, str(new_min_reached), cur_l1, None, None, None, tdcvmin, tdcvmax, td_tr_l1, tdtrmin, tdtrmax])
            
            with open(g_d_loss_path, 'a', newline='') as csvfile1:
                writer1 = csv.writer(csvfile1, dialect=csv.excel)
                writer1.writerow([step, g_loss, d_loss1, d_loss2])

                    
        if step % 500 == 0 or step == (end_iter - 1):

            ctime = time.time()
        
            print("time at iteration {} is {}".format(step, (ctime - start_time)))
            # Save the model


            #CHANGE
            #save_path = saver.save(sess, model_save_path)
            
            #TO ADD SAVE MODEL....
            #print("Model saved in file: %s" % save_path)




#save curmin to file
#curmin should be saved in outer directory
output = open(cur_min_path, 'wb')
pickle.dump(cur_min, output,2)
output.close()

