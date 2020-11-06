#%%

# =============================================================================
# Imports
# =============================================================================

import pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os


# =============================================================================
# Matplotlib style params
# =============================================================================

plt.rcParams.update(mpl.rcParamsDefault)
mpl_params = {
        'font.family' : 'Arial',
        'font.size' : 8, # Science 6-10pt, Nature 5-9pt, APL minimum 8pt
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
        'ytick.minor.width' : 0.5,
        }
plt.rcParams.update(mpl_params)
plt.ion()
colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']


#==========================================================================
# Create figure
# =============================================================================
# Set figure parameters
figure_filepath = os.getcwd()+'/../figures'
figure_filename = r'layer_rates_comparison'
width = 4 # In inches
ratio = 1.618 # A value of 2 means it's twice as wide as tall
height = width/ratio
#height = 3
fig, axes = plt.subplots(1, 3, sharex=False, sharey=True)
fig.set_size_inches(width, height)

#ax.set_yscale('log')
#ax.set_xscale('log')
axes[0].set_ylabel('Layer pair training rate')
axes[0].set_xlabel('Epoch')
axes[1].set_xlabel('Epoch')
for ax in axes:
    ax.set_xlim([0, 200])
    ax.set_ylim([1e-8, 1e-3])
    ax.set_yscale('log')
    ax.grid()

# =============================================================================
# Scatter/line plot
# =============================================================================

mpl_style_args = dict(
#        linestyle = '-',
#        linewidth = 1.0,
        marker = '.',
        markersize = 3,
#        label = 'test',
#        fillstyle='full',
#        markeredgecolor='red',
#        markeredgewidth=0.0,
        )

filepath = os.getcwd()+'/../data'
filenames = [r'single_rate_layered.pickle',
             r'perlayer_rates_layered.pickle',
             r'ourtop_1e5replaced.pickle']
learning_rates = [[.02, .02, .02, .02],
                  [1, 1, 1, 1], # In early trials, layer rate data was not scaled
                                #  by the learning rate before saving; thus, it
                                #  is necessary to scale the first and third datasets,
                                #  but not the second, by the learning rate used to
                                #  train them.
                  [.02, .02, .02, .02]]

titles = ['Layered, global\nlearning rate',
          'Layered, per-layer\nrates',
          'Our topology,\np = 7.56%']
labels = ['Input to Hidden Layer 1 ($\Delta \hat{w}^1(b)$)',
          'Hidden Layer 1 to Hidden Layer 2 ($\Delta \hat{w}^2(b)$)',
          'Hidden Layer 2 to Hidden Layer 3 ($\Delta \hat{w}^3(b)$)',
          'Hidden Layer 3 to Output ($\Delta \hat{w}^4(b)$)']

n_avg = 5

#fig.suptitle('Training rates of layer pairs with different topologies\n')
axes[1].set_xlabel('Epoch')
axes[0].set_ylabel('Layer pair training rate')
for [ax, lr, filename, title] in zip(axes, learning_rates, filenames, titles):
    with open(os.path.join(filepath, filename), 'rb') as F:
        results = pickle.load(F)
    layer_rates = results['layer rates']
    ax.set_yscale('log')
    ax.set_xlim([0, 200])
    ax.set_ylim([1e-9, 1e-3])
    ax.set_title(title)
    for pair, i in zip(layer_rates, range(len(layer_rates))):
        averaged_rates = []
        for j in range(n_avg, len(pair)-n_avg):
            averaged_rates.append(lr[i]*np.mean(pair[j-n_avg:j+n_avg+1]))
        ax.plot([.2*j for j in range(len(averaged_rates))], averaged_rates, '.',
                 color=colors[i],
                 label=labels[i],
                 **mpl_style_args)

axes[1].plot([], [], '.', color=colors[0], **mpl_style_args)
axes[1].plot([], [], '.', color=colors[1], **mpl_style_args)
axes[1].plot([], [], '.', color=colors[2], **mpl_style_args)
axes[1].plot([], [], '.', color=colors[3], **mpl_style_args)

l = axes[1].legend(['Input to H1',
                  'H1 to H2',
                  'H2 to H3',
                  'H3 to Output'], loc='lower center',
                  framealpha=.9, ncol=2, prop={'size': 6})

axes[1].set_zorder(1)
    
# =============================================================================
# Save data
# =============================================================================
    
fig.savefig(os.path.join(figure_filepath, figure_filename + '.pdf'), bbox_inches='tight', dpi=300)
