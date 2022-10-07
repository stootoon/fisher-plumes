import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
plt.style.use("default")
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

named_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def setup_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_tick_params(length=2, which="major")
    ax.yaxis.set_tick_params(length=2, which="major")
    ax.xaxis.set_tick_params(length=1, which="minor")
    ax.yaxis.set_tick_params(length=1, which="minor")
    ax.tick_params(labelsize=6)

color2hex  = lambda c: named_colors[c] if c[0] != "#" else c
hex2rgb_   = lambda h: [int(h[i:i+2],16)/255. for i in range(1,len(h),2)] + [1]
hex2rgb    = lambda h: hex2rgb_(color2hex(h))
set_alpha_ = lambda c,a: list(list(c[:3]) + [a])
set_alpha  = lambda c, a: set_alpha_(c if type(c) in [list,tuple] else hex2rgb(c), a)

spines_off = lambda ax, which=['top','right']: [ax.spines[w].set_visible(False) for w in which]

def pplot(*args, **kwargs):
    plt.plot(*args, **kwargs)
    spines_off(plt.gca())
    
