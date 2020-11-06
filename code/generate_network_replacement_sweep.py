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
        'ytick.minor.width' : 0.5,
        }
plt.rcParams.update(mpl_params)
plt.ion()
colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

#==========================================================================
# Create figure - DESCRIBEME
# =============================================================================
# Set figure parameters
figure_filepath = os.getcwd()+'/../figures'
figure_filename = r'network_replacement_sweep'
width = 4 # In inches
ratio = 1.618 # A value of 2 means it's twice as wide as tall
height = width/ratio
#height = 3
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(width, height)

# =============================================================================
# Scatter/line plot
# =============================================================================

mpl_style_args = dict(
        linewidth = 1.0,
        marker = '.',
        markersize = 3,
        )


total_edges = (28**2+3*500+10)**2 # between all pairs of neurons in network
total_edges -= (28**2)**2 + 10**2 # exclude pairs within input and output layers
total_edges -= 3*500 # exclude self-connections within hidden layers
total_edges *= .5
N_o = (28**2)*500 + 2*500**2 + 500*10 # between adjacent layers
N_o += 1.5*(500**2 - 500) # within hidden layers
N_l = total_edges - N_o # candidate connections that have not been made

def calc_p(n):
    global N_o
    global N_l
    return N_l-N_l*((N_l/(N_l+N_o))+(N_o/(N_l+N_o))*(1-((N_o+N_l)/(N_o*N_l)))**n)**(1/N_o)

filepath = os.getcwd()+'/../data'
filename = r'num_connections_sweep.pickle'
with open(os.path.join(filepath, filename), 'rb') as F:
    Results = pickle.load(F)

ax1.plot([calc_p(n) for n in Results['number of connections']],
          Results['training error'],
          '.', color='k',
          **mpl_style_args)
ax1.set_xscale('log')
ax1.set_title('Error rate after first epoch')
ax1.set_ylabel('Error rate (%)')
ax2.set_xlabel('p')
ax2.set_ylabel('Layer pair training rate')

layer_rates = [[] for i in range(4)]
spreads = []
for epoch in Results['per-layer error']:
    layer_rate = [np.sum(pair) for pair in epoch]
    for i in range(4):
        layer_rates[i].append(layer_rate[i])
    spreads.append(layer_rate[-1]/layer_rate[0])

ax2.set_title('Correction to weights of varying depth during first epoch')
ax2.set_xscale('log')
ax2.set_yscale('log')
labels = ['Input to H1',
          'H1 to H2',
          'H2 to H3',
          'H3 to Output']
for i in range(4):
    ax2.plot([calc_p(n) for n in Results['number of connections']],
              layer_rates[i],
              '.', color=colors[i], label=labels[i],
              **mpl_style_args)
ax2.legend(loc='lower right', framealpha=.9, ncol=4, prop={'size': 6}, columnspacing=.2, handletextpad=.1, handlelength=1)

ax1.grid()
ax2.grid()

plt.tight_layout()

# =============================================================================
# Save data
# =============================================================================
    
fig.savefig(os.path.join(figure_filepath, figure_filename + '.pdf'), dpi=300)
