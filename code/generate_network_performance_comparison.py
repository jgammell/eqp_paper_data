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

      
#==========================================================================
# Create figure 
# =============================================================================
# Set figure parameters
figure_filepath = os.getcwd()+r'/../figures'
figure_filename = r'network_comparison'
width = 4 # In inches
ratio = 1.618 # A value of 2 means it's twice as wide as tall
height = width/ratio
#height = 3
fig, ax = plt.subplots()
#fig, axs = plt.subplots(2,1, sharex = False, sharey = False) # Multi-subplot 
fig.set_size_inches(width, height)

#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error rate on training dataset (%)')
ax.set_xlim([0, 250])
ax.set_ylim([0, 8])
#plt.tight_layout()

#============================================================================
# Load data 
# =============================================================================

## Load data
filepath = os.getcwd()+'/../data'
with open(os.path.join(filepath, r'single_rate_layered.pickle'), 'rb') as F:
    Results = pickle.load(F)
    onerate_training_error = [100*e for e in Results['training error'][:250]]
    onerate_test_error = [100*e for e in Results['test error'][:250]]
with open(os.path.join(filepath, r'perlayer_rates_layered.pickle'), 'rb') as F:
    Results = pickle.load(F)
    multirate_training_error = [100*e for e in Results['training error'][:250]]
    multirate_test_error = [100*e for e in Results['test error'][:250]]
with open(os.path.join(filepath, r'ourtop_1e5replaced.pickle'), 'rb') as F:
    Results = pickle.load(F)
    replaced_training_error = [100*e for e in Results['training error'][:250]]
    replaced_test_error = [100*e for e in Results['test error'][:250]]
epochs = [epoch for epoch in range(1, 251)]

# =============================================================================
# Scatter/line plot
# =============================================================================

mpl_style_args = dict(
        linewidth = 1.0
        )
ax.plot(epochs, onerate_training_error,
        color = colors[0],
        linestyle = '-',
        label = 'Layered topology, global learning rate',
        **mpl_style_args)
ax.plot(epochs, onerate_test_error,
        color = colors[0],
        linestyle = '--',
        **mpl_style_args)
ax.plot(epochs, multirate_training_error,
        color = colors[1],
        linestyle = '-',
        label = 'Layered topology, per-layer rates',
        **mpl_style_args)
ax.plot(epochs, multirate_test_error,
        color = colors[1],
        linestyle = '--',
        **mpl_style_args)
ax.plot(epochs, replaced_training_error,
        color = colors[2],
        linestyle = '-',
        label = 'Our topology, p = 7.56%',
        **mpl_style_args)
ax.plot(epochs, replaced_test_error,
        color = colors[2],
        linestyle = '--',
        **mpl_style_args)

ax.grid()
plt.legend()

plt.tight_layout()

# =============================================================================
# Save data
# =============================================================================
    
fig.savefig(os.path.join(figure_filepath, figure_filename + '.pdf'), dpi=300)
