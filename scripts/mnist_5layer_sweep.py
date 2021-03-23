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
cmap = mpl.cm.get_cmap('plasma')
colors = [cmap(i/7) for i in np.arange(6)]

figure_filepath = os.path.join(os.getcwd(), '..', 'output')
figure_filename = r'mnist_5layer_sweep'
width = 4
ratio = 1
height = width/ratio
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(width, height)

p = [0]+[pp for pp in np.logspace(-4, 0, 30)][:-1]
training_errors = []
test_errors = []
rates_means = []
rates_stdevs = []
log_spread = []

for pp in p:
    training_error = []
    test_error = []
    rates_mean = []
    rates_stdev = []
    for epoch in np.arange(10):
        with open(os.path.join(os.getcwd(), '..', 'results', 'mnist_5layer_sweep', 'trial_p%.06f'%(pp), 'e%d.pickle'%(epoch)), 'rb') as F:
            Results = pickle.load(F)
            training_error.append(100*float(Results['training error']))
            test_error.append(100*float(Results['test error']))
            rates_mean.append(np.nanmean(.015*np.array([r for r in Results['per-layer rates']]), axis=0))
            rates_stdev.append(np.std(.015*np.array([r for r in Results['per-layer rates']]), axis=0))
    training_errors.append(training_error[-1])
    test_errors.append(test_error[-1])
    rates_means.append(np.mean(rates_mean, axis=0))
    rates_stdevs.append(np.mean(rates_stdev, axis=0))
    log_spread.append(np.std(np.log10(rates_mean)))
    
mpl_style_args = dict(
        linewidth = 1.0
        )
ax[0].set_xlabel('p')
ax[0].set_ylabel('Error rate (%) after 10 epochs')
ax[0].set_xscale('log')
ax[0].plot(p, training_errors, color='blue', label='Training error after 10 epochs', **mpl_style_args)
ax[0].plot(p, test_errors, '--', color='blue', label='Test error after 10 epochs', **mpl_style_args)
a0tx = ax[0].twinx()
a0tx.plot(p, log_spread, color='red', label='Std. dev. of log10(spread), mean over 10 epochs', **mpl_style_args)
a0tx.set_ylabel('Mean log-spread over 10 epochs')
a0tx.spines['left'].set_color('blue')
a0tx.spines['right'].set_color('red')
ax[0].tick_params(axis='y', colors='blue')
a0tx.tick_params(axis='y', colors='red')
print('Correlation coefficient between stdev-log and training error:', (np.corrcoef(training_errors, log_spread)[0, 1])**2)

# mpl_style_args = dict(
#         marker = '.',
#         markersize = 3,
#         linestyle='None'
#         )
ax[1].set_xlabel('p')
ax[1].set_ylabel('RMS Correction')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
for i in range(len(rates_means[0])):
    pair_mean = np.array([om[i] for om in rates_means])
    pair_std = np.array([os[i] for os in rates_stdevs])
    ax[1].plot(p, pair_mean, color=colors[i], **mpl_style_args)
    #ax[1].fill_between(p, pair_mean-pair_std, pair_mean+pair_std, color=colors[i], alpha=.2)
    ax[1].set_ylim([1e-7, 1e-3])

ax[0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
a0tx.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
ax[1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
ax[1].legend(['INPUT-H1', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-OUTPUT'], loc='lower center', ncol=3, markerscale=.5, labelspacing=.25, columnspacing=.5, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(figure_filepath, figure_filename+'.pdf'), dpi=300)