import numpy as np
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
import os
import imageio

from shape_dataset_code.plot_3d import *

#saved = "models/bu_more_data/cv_generated_iter_3300"
#saved = "models/model_177/cv_generated_iter_1860"
saved = "models/bu continued model/cv_generated_iter_1460"
if not os.path.isdir(saved):
    os.mkdir(saved)
data_3d = np.load(saved + ".npy")

print("min:{} max:{} mean:{}".format(np.min(data_3d),np.max(data_3d),np.mean(data_3d)))
print("min:{} max:{} mean:{}".format(np.min(data_3d),np.max(data_3d),np.mean(data_3d)))

data_3d[(data_3d >= .6)] = 1
data_3d[(data_3d < .4)] = 0
print(np.unique(data_3d))
print(data_3d[0].shape)

x,y,z = np.meshgrid(np.linspace(-1,1,64), np.linspace(-1,1,64), np.linspace(-1,1,64))
#for i in range(0,data_3d.shape[0]):
for i in [3,14,15]:
    plot_3d(data_3d[i], x, y, z, savename=saved + "/{}.png".format(i), display=False, save=True)