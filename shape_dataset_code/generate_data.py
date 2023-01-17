import numpy as np
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
import os
import imageio
from get_fig import *
#import plotly.express as px

from shape_dataset_helper_functions import *
from shape_dataset_code.plot_3d import *
import gc


def generate_shape_data(n_shapes,rot_z = True, rot_y = True, xres = 64, yres = 64, zres = 64,
                    base_output_dir = '',savedir ='shape_data'):

    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    for i in range(1000):
        new_dir = savedir + '/' + "batch_{}".format(i)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
            savedir = new_dir
            break
        if i == 999:
            raise Exception("Ran out of batch files or some other error may need to expand length of for loop")
        
    print("savedir: {}".format(savedir))
    
    #mesh x,y,z
    x,y,z = np.meshgrid(np.linspace(-1,1,xres), np.linspace(-1,1,yres),
                    np.linspace(-1,1,zres))
    
    shape_type_list = ['cuboid', 'ellipsoid','cylinder', 'octahedron']
    shape_list_3d = []; shape_list_2d = []; shape_info_list = []

    for i in range(n_shapes):

        x_temp = x; y_temp = y; z_temp = z;

        #constrain shapes to be asymmetric and generate symmetric shapes 
        #as a different label
        
        #RANDOM ROTATION
        #list z rotations
        z_rot_list = [22.5 * i for i in range(8)]
        #list y rotations
        y_rot_list = [22.5 * i for i in range(8)]
        
        #if random rotation z
        if rot_z:
            #randomly select z rotation from list
            zind = np.random.randint(0,8)
            z_rot = z_rot_list[zind]
            
            #z_rot = 180
            print("z rot is {}".format(z_rot))
            x_temp, y_temp = rotate_counter(x_temp, y_temp, z_rot)
    
        #if random rotation y
        if rot_y:
            #randomly select y rotation from list
            yind = np.random.randint(0,8)
            y_rot = y_rot_list[yind]
            #y_rot = 45
            print("y rot is {}".format(y_rot))
            x_temp, z_temp = rotate_counter(x_temp,z_temp, y_rot)
            
        #SELECT SHAPE
        n_shape_type = len(shape_type_list)
        shape_ind = np.random.randint(0,n_shape_type)
        shape = shape_type_list[shape_ind]
        #shape = 'octahedron'
        print("shape is {}".format(shape))



        res_arr = np.linspace(-1,1,64)
        
        #randomly select shape parameters
        #obtain 3d shape matrix (vals) using generated random params
            
        #if cuboid
        if shape == 'cuboid':
            height = np.random.randint(30,55)
            zmin, zmax = split(height, zres)

            #height random between 49 and 16
            wh = round(height / 1.5)
            wl = round(height/5)
            
            #width, length random from  round(height/1.5) to round(height/5)
            width = np.random.randint(wl, wh+1)
            xmin, xmax = split(width,xres)
            
            length = np.random.randint(wl, wh+1)
            ymin, ymax = split(length, xres)

            vals = within_cuboid(x=x_temp, y=y_temp, z=z_temp,
                xmax=xmax, xmin=xmin, ymax=ymax, 
                ymin=ymin, zmax=zmax, zmin=zmin)
            
            print("cuboid vals zmin:{} zmax:{}, xmin:{}, xmax:{} ymin:{} ymax:{}".format(
                    zmax, zmin, xmin, xmax, ymin, ymax))
            #2 position, 2 rotation, 1 shape type, 6 shape dims = length 11
            shape_data = np.array([z_rot, y_rot, 0, xmin, xmax, 
                                   ymin, ymax, zmin, zmax ])
            
            
        #if ellipsoid
        elif shape == 'ellipsoid':
            height = np.random.randint(35,55)
            
            #height random between 49 and 16
            wh = round(height / 1.5)
            wl = round(height/5)
            
            #width, length random from  round(height/1.5) to round(height/5)
            width = np.random.randint(wl, wh+1)
            length = np.random.randint(wl, wh+1)

            height = (height/zres)
            width = (width/xres)
            length = (length/yres)

            vals = within_ellipsoid(x=x_temp, y=y_temp,
                z = z_temp, xlim = width,ylim = length,
                zlim = height)
            

            print("ellipsoid vals height:{}, width:{}, length:{}".format(
                    height, width, length))
            shape_data = np.array([z_rot, y_rot, 1, width, length, height, np.nan, np.nan, np.nan])
            
        #if cylinder
        elif shape == 'cylinder':
            prod = 1
            count = 0
            while prod > .18:
                #height random between 16 and 49
                height = np.random.randint(32,48)
                zmin, zmax = split(height, zres)
                #width, length random from  8 to 33
                width = np.random.randint(16,32)
                length = np.random.randint(16,32)
                #halves but remember that xlim is the radius
                width = (width/xres)
                length = (length/yres)
                prod = zmax * width * length
                count += 1
                if count > 10000:
                    raise Exception("infinite while loop")
                    break
                
                
            
            print("cylinder vals high:{} low:{}, width:{}, length:{}".format(
                                    zmax, zmin, width, length))

            vals = within_cylinder(high = zmax, low=zmin,
                xlim = width, ylim=length, 
                x=x_temp,y=y_temp,z=z_temp)
            
            shape_data = np.array([z_rot, y_rot, 2, width, length, zmin, zmax, np.nan, np.nan])
            
        #if octahedron
        elif shape == 'octahedron':
            #height random between 49 and 16
            height = np.random.randint(34,58)
            #width, length random from  33 to 8
            width = np.random.randint(17,30)
            length = np.random.randint(17,30)
            
            height= (2*height/zres)
            width = (2*width/xres)
            length = (2*length/yres)
            
            #x_len, y_len, height are total width,length, height
            vals = within_octahedron(x_len=width, y_len = length,
                    height = height, x=x_temp,y=y_temp,z=z_temp)
            
            print("octahedron vals height:{}, width:{}, length:{}".format(
                    height, width, length))
            shape_data = np.array([z_rot, y_rot, 3, width, length, height, np.nan, np.nan, np.nan])
        
        else:
            raise Exception("shape not recognized")
        
        print("vals shape {}".format(vals.shape))
        print("vals sum {}".format(np.sum(vals)))
        
        #SHAPE SHIFTING
        #shift shape to floor
        vals = shift_shape_to_floor(vals)

        #if randomly shift position
            #randomly shift position
        vals, position = random_position_shift(vals)
        
        
        image_save_name = "/{}.png".format(str(i))
        #save 2d image in savedir
        pfig(vals,x,y,z, savename=savedir + image_save_name)
        gc.collect()
        
        #load, crop, remove old, save 2d image
        im = imageio.imread(savedir + image_save_name)
        im_crop = im[54:310,45:301].copy()
        m,n,c = im_crop.shape
        assert(m==n and n==256)
        os.remove(savedir+ image_save_name)
        plt.imsave(savedir + image_save_name, im_crop)
        
        #append 2d matrix to 2d image list
        shape_list_2d.append(im_crop)
        
        assert(np.sum(vals) > 100)
        #append vals to 3d shape list
        shape_list_3d.append(vals)
        
        #store position, rotation, shape type, shape dims in array
        #2 position, 2 rotation, 1 shape type, 6 shape dims = length 11
        shape_data = np.concatenate([np.array(position), shape_data])
        assert(shape_data.shape[0] == 11)
        
        shape_info_list.append(shape_data)
        
    #stack shape info, 2d, and 3d images into arrays
    data_1d = np.stack(shape_info_list, axis = 0)
    data_2d = np.stack(shape_list_2d, axis = 0)
    data_3d = np.stack(shape_list_3d, axis = 0)
    
    print("data 1d shape: {}".format(data_1d.shape))
    print("data 2d shape: {}".format(data_2d.shape))
    print("data 3d shape: {}".format(data_3d.shape))
    
    np.save(savedir + "/data_1d.npy", data_1d)
    np.save(savedir + "/data_2d.npy", data_2d)
    np.save(savedir + "/data_3d.npy", data_3d)


generate_shape_data(n_shapes=16, rot_z=True, rot_y=True, xres=64, yres=64, zres=64, base_output_dir='')
        
    
    
    
