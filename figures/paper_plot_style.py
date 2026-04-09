"""Shared publication style for all paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
FORMAT = 'pdf'
FONT_SIZE = 10
FIG_DIR = 'figures'

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

COLORS = plt.cm.tab10.colors
C_DSC = COLORS[0]      # blue
C_DISS = COLORS[3]     # red
C_LCDM = '#333333'     # dark gray
C_AI = COLORS[3]       # red
C_PLANCK = COLORS[3]   # red
C_GAUSS = '#333333'    # black dashed


def save_fig(fig, name, fmt=FORMAT):
    path = f'{FIG_DIR}/{name}.{fmt}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {path}')
