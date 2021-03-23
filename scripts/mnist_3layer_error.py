#%%
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

plt.rcParams.update(mpl.rcParamsDefault)
mpl_params = {
        'font.family' : 'Arial',
        'font.size' : 8,
        'mathtext.fontset' : 'dejavuserif',
        'ytick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'lines.linewidth': 0.5,
        'axes.linewidth' : 0.5,
        'xtick.major.width' : 0.5,
        'ytick.major.width' : 0.5,
        'xtick.minor.width' : 0.5,
        'ytick.minor.width' : 0.5
        }
plt.rcParams.update(mpl_params)
plt.ion()
colors = ['#1b9e77', '#d95f02', 'k']

figure_filepath = os.path.join(os.getcwd(), '..', 'output')
figure_filename = r'mnist_3layer_error'
width = 4
ratio = 1.618
height = width/ratio
fig, ax = plt.subplots()
fig.set_size_inches(width, height)

num_epochs = 250

folders = ['mnist_3layer_mlff', 'mnist_3layer_sw', 'mnist_3layer_swni']

mlff = {'epochs': [], 'training': [], 'test': []}
with open(os.path.join(os.getcwd(), '..', 'results', 'mnist_3layer_mlff2020', 'mnist_3layer_mlff.pickle'), 'rb') as F:
    Results = pickle.load(F)
    mlff['training'] = [100*e for e in Results['training error'][:num_epochs]]
    mlff['test'] = [100*e for e in Results['test error'][:num_epochs]]
mlff['epochs'] = np.arange(1, num_epochs+1)
perlayer = {'epochs': [], 'training': [], 'test': []}
with open(os.path.join(os.getcwd(), '..', 'results', 'mnist_3layer_perlayer2020', 'mnist_3layer_perlayer.pickle'), 'rb') as F:
    Results = pickle.load(F)
    perlayer['training'] = [100*e for e in Results['training error'][:num_epochs]]
    perlayer['test'] = [100*e for e in Results['test error'][:num_epochs]]
perlayer['epochs'] = np.arange(1, num_epochs+1)
swni = {'epochs': [], 'training': [], 'test': []}
for epoch in range(num_epochs):
    try:
        with open(os.path.join(os.getcwd(), '..', 'results', 'mnist_3layer_swni', 'e%d.pickle'%(epoch)), 'rb') as F:
            Results = pickle.load(F)
        swni['epochs'].append(epoch+1)
        swni['training'].append(100*float(Results['training error']))
        swni['test'].append(100*float(Results['test error']))
    except:
        break

mpl_style_args = dict(
        linewidth = 1.0
        )
ax.plot(mlff['epochs'], mlff['training'],
        color = colors[0],
        linestyle = '-',
        label = 'MLFF, global learning rate',
        **mpl_style_args)
ax.plot(mlff['epochs'], mlff['test'],
        color = colors[0],
        linestyle = '--',
        **mpl_style_args)
ax.plot(swni['epochs'], swni['training'],
        color = colors[1],
        linestyle = '-',
        label = 'SW, p=10%',
        **mpl_style_args)
ax.plot(swni['epochs'], swni['test'],
        color = colors[1],
        linestyle = '--',
        **mpl_style_args)
ax.plot(perlayer['epochs'], perlayer['training'],
        color = colors[2],
        linestyle = '-',
        label = 'MLFF, per-layer rates',
        **mpl_style_args)
ax.plot(perlayer['epochs'], perlayer['test'],
        color = colors[2],
        linestyle = '--',
        **mpl_style_args)

ax.grid()

ax.set_xlim([0, 250])
ax.set_ylim([0, 8])
ax.set_xlabel('Epoch')
ax.set_ylabel('Error rate (%)')
#ax.set_title('3 500-neuron hidden layers on MNIST')
plt.legend()

plt.tight_layout()

fig.savefig(os.path.join(figure_filepath, figure_filename + '.pdf'), dpi=300)