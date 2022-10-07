import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
plt.style.use("default")
from matplotlib import colors as mcolors
from matplotlib.gridspec   import GridSpec
from matplotlib import cm

import fisher_plumes_fig_tools as fpft
#import crick_fisher_tools as cft

dist2col = lambda d, dcol_scale = 120000, cmap = cm.cool_r: cmap(d/dcol_scale)

def plot_correlations(rho, 
                      xl = (-50, 50),
                      slices = {"all":slice(1,10000), "first":slice(1,2),   "second":slice(10,11)},
                      cols   = {"all":cm.gray(0.4),   "first":cm.cool(1.0), "second":cm.cool(0.2)},
                      n_rows = 1,
                      plot_order = None, 
                      figsize=None,
):

    dists    = np.array(sorted(list(rho.keys()))) 
    rho      = {d:rho[d][0] for d in dists} # [0] to take the raw data
    rho_mean = {k:np.array([np.mean(np.sum(rho[d][slc,:],axis=0)) for d in dists]) for k, slc in slices.items()}
    rho_std  = {k:np.array([ np.std(np.sum(rho[d][slc,:],axis=0)) for d in dists]) for k, slc in slices.items()}
    rho_pc   = {k:{pc:np.array([np.percentile(np.sum(rho[d][slc,:],axis=0), pc) for d in dists]) for pc in [5,50,95]} for k, slc in slices.items()}

    n_slices = len(slices)
    n_panels = n_slices + 1
    n_cols   = int(np.ceil(n_panels/n_rows))
    plt.figure(figsize=figsize if figsize else (3*n_cols, 3 * n_rows))
    ax = []
    gs = GridSpec(n_rows, n_cols)
    if plot_order is None: plot_order = list(slices.keys())
    for i, k  in enumerate(plot_order):
        irow = int(i // n_cols)
        icol = int(i %  n_cols)
        ax.append(plt.subplot(gs[irow, icol]))
        sc= 1#max(rho_mean[k])
        y = rho_mean[k]/sc
        ee= rho_std[k]/sc
        dists_mm = dists/1000
        plt.fill_between(dists_mm, y-ee, y+ee, color=fpft.set_alpha(mpl.colors.to_rgba(cols[k]),0.2));
        fpft.pplot(dists_mm, y , "o-", markersize=4,color=cols[k]);
        plt.xlabel("Distance (mm)")
        (icol == 0) and plt.ylabel("Correlation")
        plt.title(k)
        plt.xlim(xl)
    
    ax.append(plt.subplot(gs[int(n_slices//n_cols), int(n_slices % n_cols)]))
    for i, k in enumerate(slices):
        fpft.pplot(dists_mm, rho_mean[k]/max(rho_mean[k]), "-", markersize=4, color=cols[k], label=k);
        plt.xlim(xl)
    plt.legend(frameon=False, fontsize=8)
    plt.xlabel("Distance (mm)")        
    plt.title("Overlayed and Scaled")
    plt.tight_layout(w_pad=0)
    return ax
