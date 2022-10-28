import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
plt.style.use("default")
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import TransformedBbox, Bbox

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
def expand_lims(lims, factor):
    d = (lims[1] - lims[0])*factor
    m = (lims[1] + lims[0])/2
    return [m-d/2, m+d/2]


def label_axes(ax_list, labs, dx=0, dy=0, x = None, y = None,
               align_x = [], align_x_fun = np.mean,
               align_y = [], align_y_fun = np.mean,
               *args, **kwargs):
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    trans    = fig.transFigure
    itrans   = trans.inverted()
    h = []
    x_vals = []
    y_vals = []
    for i, (ax, lab) in enumerate(zip(ax_list, labs)):
        bb = ax.get_tightbbox(renderer)
        bb = TransformedBbox(bb, itrans)
        dxi = dx[i] if hasattr(dx, "__len__") else dx
        dyi = dy[i] if hasattr(dy, "__len__") else dy
        xi = bb.x0 + dxi if x is None else x[i]
        yi = bb.y1 + dyi if y is None else y[i]
        x_vals.append(xi)
        y_vals.append(yi)
        h.append(fig.text(xi, yi, lab, *args, transform=trans, **kwargs))

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    if len(align_x):
        for grp in align_x:
            x_vals[grp] = align_x_fun(x_vals[grp])
            
    if len(align_y):
        for grp in align_y:
            y_vals[grp] = align_y_fun(y_vals[grp])

    for hi, x,y in zip(h, x_vals, y_vals):
        hi.set_position((x,y))

    
            
            
    
    
