#%%
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import torch

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

cmap = mpl.cm.get_cmap('plasma')
colors = [cmap(i/4) for i in np.arange(4)]

figure_filepath = os.path.join(os.getcwd(), '..', 'output')
figure_filename = r'mnist_3layer_rates'
width = 4
ratio = 2
height = width/ratio

fig, axes = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(width, height)

folders = ['mnist_3layer_mlff', 'mnist_3layer_swni', 'mnist_3layer_mlffpl']
mpl_style_args = dict(
        marker = '.',
        markersize = 3,
        linestyle='none'
        )

mlff_3layers = {'means': [], 'stdevs': []}
swni_3layers = {'means': [], 'stdevs': []}
mlffpl_3layers = {'means': [], 'stdevs': []}

for folder, out, lr in zip(folders, [mlff_3layers, swni_3layers, mlffpl_3layers], [.02, .02, [.128, .032, .008, .002]]):
    for epoch in range(100):
        try:
            with open(os.path.join(os.getcwd(), '..', 'results', folder, 'e%d.pickle'%(epoch)), 'rb') as F:
                Results = pickle.load(F)
        except:
            print(folder, epoch)
            continue
        if type(lr) == float:
            out['means'].append(np.mean(lr*np.array(Results['per-layer rates']), axis=0))
            out['stdevs'].append(np.std(lr*np.array(Results['per-layer rates']), axis=0))
        elif type(lr) == list:
            val = np.mean(np.array(Results['per-layer rates']), axis=0)
            for idx, lrr in enumerate(lr):
                val[idx] *= lrr
            out['means'].append(val)
        else:
            assert False

axes[1], axes[2] = axes[2], axes[1]
titles = ['MLFF, global\nlearning rate', 'SW, p=10%\n', 'MLFF, per-layer\nrates']
for ax, title, data in zip(axes, titles, [mlff_3layers, swni_3layers, mlffpl_3layers]):
    ax.set_yscale('log')
    ax.set_xlim([1, 100])
    ax.set_title(title)
    for i in range(len(data['means'][0])):
        pair_mean = np.array([om[i] for om in data['means']])
        pair_std = np.array([os[i] for os in data['stdevs']])
        ax.plot(np.arange(1, len(pair_mean)+1), pair_mean, color=colors[i], **mpl_style_args)
        #ax.fill_between(np.arange(1, 101), pair_mean-pair_std, pair_mean+pair_std, color=colors[i], alpha=.2)
        ax.grid()
        ax.set_ylim([5e-7, 1e-3])
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([1e-3, 1e-4, 1e-5, 1e-6])
        ax.set_xlabel('Epochs')
axes[0].set_ylabel('RMS Correction')
fig.legend(['INPUT-H1', 'H1-H2', 'H2-H3', 'H3-OUTPUT'], loc='center', bbox_to_anchor=(1.1,.5), ncol=1, markerscale=2, labelspacing=.25, columnspacing=.5)

plt.tight_layout()

fig.savefig(os.path.join(figure_filepath, figure_filename + '.pdf'), dpi=300, bbox_inches='tight')





