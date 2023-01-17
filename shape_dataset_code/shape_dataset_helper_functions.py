import numpy as np
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
import os
import imageio


def within_ellipsoid(x,y,z, xlim,ylim,zlim):
    vals = np.zeros(x.shape)
    total = (x/xlim)**2 + (y/ylim)**2 + (z/zlim)**2
    vals[(total <1)] = 1
    return vals


def within_cuboid(x,y,z, xmax, xmin,
                 ymax, ymin, zmax, zmin):
    vals = np.zeros(x.shape)
    vals[(x < xmax) & (x > xmin)\
         & (y < ymax) & (y > ymin) & (z < zmax) & (z > zmin)] = 1
    return vals


def within_range(mat, high, low):
    vals = np.zeros(mat.shape)
    vals[(mat < high) & (mat > low)] = 1
    return vals
    

def within_elipse(x,y, xlim, ylim):
    vals = np.zeros(x.shape)
    total = (x/xlim)**2 + (y/ylim)**2
    vals[(total <1)] = 1
    return vals


def within_cylinder(high, low, xlim, ylim, x,y,z):
    #get within range
    wrange = within_range(z,high,low)
    #get within elipse
    welipse = within_elipse(x,y,xlim,ylim)
    
    vals = np.zeros(wrange.shape)
    vals[(wrange == 1) & (welipse == 1)] = 1
    return vals


# A utility function to calculate area
# of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
def area(x1, y1, x2, y2, x3, y3):
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)))/2


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):
 
    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)
     
    vals = np.zeros(x.shape)
    
    Asum = A1 + A2 + A3
    
    isclose = np.isclose(A,Asum)
    
    vals[(isclose == True)] = 1
    return vals


def within_pyramid(x_len, y_len, height, x,y,z):
    
    x1 = -x_len/2
    x2 = 0
    x3 = x_len/2

    y1 = -y_len/2
    y2 = 0
    y3 = y_len/2

    z1 = 0
    z2 = height/2
    z3 = 0
    
    v1 = isInside(x1, z1, x2, z2, x3, z3, x, z)
    v2 = isInside(y1, z1, y2, z2, y3, z3, y, z)
    v3 = np.zeros(x.shape)
    v3[(v1 == 1) & (v2 == 1)] = 1
    
    return v3


def within_octahedron(x_len, y_len, height, x,y,z):

    x1 = -x_len/2
    x2 = 0
    x3 = x_len/2

    y1 = -y_len/2
    y2 = 0
    y3 = y_len/2

    z1 = 0
    z2 = height/2
    z3 = 0
    
    z22 = -height/2
    
    v1 = isInside(x1, z1, x2, z2, x3, z3, x, z)
    v2 = isInside(y1, z1, y2, z2, y3, z3, y, z)
    
    v3 = isInside(x1, z1, x2, z22, x3, z3, x, z)
    v4 = isInside(y1, z1, y2, z22, y3, z3, y, z)
    
    vals = np.zeros(x.shape)
    vals[(v1 == 1) & (v2 == 1)] = 1
    vals[(v3 == 1) & (v4 == 1)] = 1
    
    return vals


def random_position_shift(vals):

    x_wid, y_wid, _ = vals.shape
    
    pos = np.where(vals == 1)
    xlocs = pos[0]
    xmin = np.min(xlocs)
    xmax = np.max(xlocs)

    ylocs = pos[1]
    ymin = np.min(ylocs)
    ymax = np.max(ylocs)
    
    xwig = x_wid - (xmax - xmin)
    ywig = y_wid - (ymax - ymin)
    
    #pick random number in xwig
    xrand = np.random.randint(0,xwig)
    xshift = xrand - xmin
    
    #pick random number in ywig
    yrand = np.random.randint(0,ywig)
    yshift = yrand - ymin
    
    sliced_loc = vals[xmin:xmax+1,ymin:ymax+1,:]
    locscopy = np.zeros(vals.shape)
    locscopy[xmin+xshift:xmax+1+xshift,ymin+yshift:ymax+1+yshift,:] = sliced_loc
    
    position = ((xmin+xshift+xmax+1+xshift)/2,(ymin+yshift+ymax+1+yshift)/2)
    
    return locscopy, position
    

def shift_shape_to_floor(vals):

    height = vals.shape[2]
    locs = np.where(vals == 1)[2]
    
    #obtain minimum l height index
    minh = np.min(locs)
    
    #slice old vals [:,:,minz:]
    vals_slice = vals[:,:,minh:]
    
    new_vals = np.zeros(vals.shape)
    
    new_vals[:,:,0:(height - minh)] = vals_slice
    
    return new_vals


def rotate_counter(x, y, theta_degrees):
    theta = (theta_degrees * np.pi) / 180
    xf = x.flatten()
    yf = y.flatten()
    
    rot_mat = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    new = np.stack([xf, yf])
    rotated = np.matmul(rot_mat, new)
    
    xnew = rotated[0]
    ynew = rotated[1]
    
    xnew = xnew.reshape(x.shape)
    ynew = ynew.reshape(x.shape)
    
    return xnew, ynew
    

def create_cylinder(thetay,thetaz, high, low, xlim, ylim, 
                    xres = 64, yres = 64, zres = 64):

    x,y,z = np.meshgrid(np.linspace(-1,1,xres), np.linspace(-1,1,yres),
                        np.linspace(-1,1,zres))

    xr, yr = rotate_counter(x, y, thetaz)
    xr, zr = rotate_counter(xr,z, thetay)
    
    vals = within_cylinder(high, low, xlim, ylim, xr, yr, zr)
    
    return x,y,z,vals


def split(num, res = None):
    if res is not None:
        arr = np.linspace(-1,1,res)
    
    #if even
    if num%2 == 0:
        if res is None:
            return -num/2, num/2 - 1
        if res is not None:
            return arr[int(res/2 + -num/2)], arr[int(res/2 + num/2 - 1)]
    
    if num%2 == 1:
        if res is None:
            return np.ceil(-num/2), np.ceil(num/2) - 1
        if res is not None:
            return arr[int(res/2 + np.ceil(-num/2))], arr[int(res/2 + np.ceil(num/2) - 1)]