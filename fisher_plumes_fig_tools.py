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
    
def plot_bivariate_gaussian(mu, C, n_sds = 1, n_points = 1000, mean_style= {"marker":"o"}, sd_style = {"linewidth":1}):
    U,s,_ = np.linalg.svd(C)
    A   = np.dot(U, np.diag(np.sqrt(s)))
    th  = np.linspace(0,2*np.pi,n_points)
    p   = [np.cos(th), np.sin(th)]
    q   = np.dot(A, p)
    hmu = plt.plot(mu[0], mu[1], **mean_style)[0]
    hsd = [plt.plot(q[0]*i + mu[0], q[1]*i + mu[1], **sd_style)[0] for i in range(1,n_sds+1)]
    return hmu, hsd

cdfplot    = lambda x, **kwargs: plt.plot(np.sort(x), np.arange(1,len(x)+1)/len(x), **kwargs)
hist_to_yx = lambda counts, bins: (np.array([0]+ list(counts) + [0]), np.array([bins[0]-(bins[1] - bins[0])/2] + list((bins[1:] + bins[:-1])/2) + [bins[-1] + (bins[1]-bins[0])/2]))
pdfplotf   = lambda x, bins = 10, **kwargs: (lambda y, x: plt.fill_between(x, y, 0*y, **kwargs))(*hist_to_yx(*np.histogram(x, bins=bins, density=True)))
pdfplot    = lambda x, bins = 10, **kwargs: (lambda y, x: plt.plot(x, y, **kwargs))(*hist_to_yx(*np.histogram(x, bins=bins, density=True)))
pdfplotl   = lambda x, bins = 10, **kwargs: (lambda y, x: plt.semilogy(x, y, **kwargs))(*hist_to_yx(*np.histogram(x, bins=bins, density=True)))
set_alpha  = lambda col, al: list(col[:3]) + [al]
vline      = lambda x, *args, **kwargs: (lambda x1, yl, *args, **kargs: (plt.plot([x1,x1],yl, *args, **kwargs), plt.gca().set_ylim(yl)))(x, plt.gca().get_ylim(), *args, **kwargs)
