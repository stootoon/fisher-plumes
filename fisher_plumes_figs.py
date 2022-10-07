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

def plot_coef1_vs_coef2(coefs, ifreq, pairs,
                        figsize=(8,3),
                        i_pos_dists_to_plot = [0,2,4],
                        dcol_scale=120000.,
                        
):
    if type(coefs) is not list: coefs = [coefs]
        
    pooled1 = {d:np.array([
        np.concatenate([coef[y1][0][:, ifreq] for coef in coefs for (y1,y2) in pairs[d]], axis = 0),
        np.concatenate([coef[y2][0][:, ifreq] for coef in coefs for (y1,y2) in pairs[d]], axis = 0)
    ]) for d in pairs} # [0] takes the raw data, not the bootstraps

    dists = np.array(sorted(pooled1.keys()))
    plt.figure(figsize=figsize)
    th = np.linspace(0,2*np.pi,1001)
    which_d = [dists[dists>0][i] for i in i_pos_dists_to_plot]
    for i, d in enumerate(which_d):
        plt.subplot(1,3,i+1)
        U,S,_ = np.linalg.svd(np.cov(pooled1[d]))
        p0 = pooled1[d][0]
        p1 = pooled1[d][1]
        p0m= np.mean(p0)
        p1m= np.mean(p1)
        plt.plot(p0,p1, "o", color=dist2col(d), markersize=1)
        plt.plot(p0m, p1m, "k+", markersize=10)
        
        for r in [1,2,3]:
            xy = U @ np.diag(np.sqrt(S)) @ [np.cos(th), np.sin(th)]
            plt.plot(r*xy[0] + p0m, r*xy[1]+p1m,":",color="k", linewidth=1)
        plt.xlim([-1.1,1.1]); plt.ylim([-1.1,1.1])
        plt.gca().set_xticks([-1,0,1]);
        plt.gca().set_yticks([-1,0,1])
        fpft.spines_off(plt.gca())
        plt.axis("square")
        (i == 0) and plt.ylabel("Coefficient at source 2")
        plt.xlabel("Coefficient at source 1")
        plt.title(f"{d/1000:g} mm")
        plt.grid(True)    

