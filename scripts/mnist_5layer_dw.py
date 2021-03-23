#%%
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

with open(os.path.join(os.getcwd(), '..', 'results', 'mnist_5layer_mlff', 'e99.pickle'), 'rb') as F:
    Results = pickle.load(F)
    mlff_dW = Results['mean dW']
with open(os.path.join(os.getcwd(), '..', 'results', 'mnist_5layer_swni', 'e99.pickle'), 'rb') as F:
    Results = pickle.load(F)
    swni_dW = Results['mean dW']

mlff_dW[mlff_dW==0] = -np.nan
swni_dW[swni_dW==0] = -np.nan
mlff_dW = np.log10(mlff_dW)
swni_dW = np.log10(swni_dW)

leg = np.ones((mlff_dW.shape[0],int(.05* mlff_dW.shape[1])))
for idx, mult in enumerate(np.linspace(-3, -5, mlff_dW.shape[1])):
    leg[idx, :] *= mult

cmap = matplotlib.cm.plasma
cmap.set_bad('white')

imsave_args = {'dpi': 500, 'cmap': cmap, 'vmin': -5, 'vmax': -3}
#imsave_args = {'dpi': 300, 'cmap': 'plasma', 'vmin': 1e-5, 'vmax': 1e-3}
plt.imsave(os.path.join(os.getcwd(), '..', 'output', 'mnist_5layer_dW__mlff.png'), mlff_dW, **imsave_args)
plt.imsave(os.path.join(os.getcwd(), '..', 'output', 'mnist_5layer_dW__swni.png'), swni_dW, **imsave_args)
plt.imsave(os.path.join(os.getcwd(), '..', 'output', 'mnist_5layer_dW__leg.png'), leg, **imsave_args)