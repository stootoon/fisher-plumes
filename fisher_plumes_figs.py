import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
plt.style.use("default")
from matplotlib import colors as mcolors
from matplotlib.gridspec   import GridSpec
from matplotlib import cm

import fisher_plumes_fig_tools as fpft
import fisher_plumes_tools as fpt

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


def plot_alaplace_fits(F, which_dists,
                       ifreq_lim = [], xl = [-0.2,0.5], which_ifreq = 1, 
                       figsize=None, vmax=None,
                       cm_heat = cm.rainbow,
                       plot_dvals = False):
    if figsize is not None:
        plt.figure(figsize=figsize)

    gs   = GridSpec(2 if plot_dvals else 1, len(which_dists)+1)
    axd  = []
    dmax = -1
    yld  = []
    for di, d in enumerate(which_dists):
        print(f"{d=:3d} @ Freq # {which_ifreq:3d}: -np.log10(p) = {-np.log10(F.pvals[d][0][which_ifreq]):1.3f}")
        plt.subplot(gs[:-1 if plot_dvals else 1,di])
        xvals = np.linspace(xl[0],xl[-1],1001)
        la_   = F.la[d][0][which_ifreq]
        mu_   = F.mu[d][0][which_ifreq]
        ypred = fpt.alaplace_cdf(la_, mu_, xvals)
        rr    = F.rho[d][0][which_ifreq]
        hdata = fpft.cdfplot(rr,color=dist2col(d), linewidth=1, label='F$_{data}$($x$)')
        hfit  = plt.plot(xvals, ypred,
                color=fpft.set_alpha(dist2col(d),0.4),
                label='F$_{fit}$($x$)',
                linewidth=1)
        fpft.vline(0, ":", color = "lightgray")
        
        fpft.spines_off(plt.gca())    
        plt.xlabel("x",labelpad=-2)
        (di == 0) and plt.ylabel("P($r_n \leq x$)")        
        plt.gca().set_yticks(np.arange(0,1.1,0.2))
        plt.xlim(xl)

        plt.legend(frameon=False, labelspacing=0, fontsize=6, loc='lower right')
        plt.title(f"{d/1000:g} mm")

        if plot_dvals:
            rx = np.array(sorted(rr))
            ex = np.arange(1, len(rr)+1)/len(rr)
            cx = fpt.alaplace_cdf(la_, mu_, rx)
            dvals = np.abs(ex - cx)
            axdi = plt.subplot(gs[-1, di])#, sharey = axd[0] if len(axd) else None)
            plt.plot(rx, dvals, color=dist2col(d), linewidth=1)
            if np.max(dvals) > dmax:
                dmax = np.max(dvals)
                yld = plt.ylim()
            fpft.spines_off(plt.gca())
            fpft.vline(0, ":", color = "lightgray")            
            plt.xlim(xl)
            axd.append(axdi)
            plt.grid(True, axis='y', linestyle=":")
            #(di != 0 ) and axdi.set_yticklabels([])
            plt.xlabel("x",labelpad=-2)
            (di == 0) and plt.ylabel("|F$_{data}$($x$) - F$_{fit}$($x$)|")

    if plot_dvals:
        [axdi.set_ylim(yld) for axdi in axd]
        #axd[-1].set_ylim(0, dmax)
            
        
    plt.subplot(gs[:,-1])

    freqs = np.arange(F.wnd)/F.wnd*F.fs
    
    pdists = np.array(sorted(F.pvals))
    dd = np.mean(np.diff(pdists))
    p = np.array([F.pvals[d][0] for d in sorted(F.pvals)]).T
    if len(ifreq_lim)==0:
        ifreq_lim = [0, p.shape[0]]
    p = p[ifreq_lim[0]:ifreq_lim[1],:]
    plt.matshow(-np.log10(p), #+np.min(p[p>0])/10),
            extent = [(pdists[0]-dd/2)/1000, (pdists[-1]+dd/2)/1000, freqs[ifreq_lim[0]]-0.5, freqs[ifreq_lim[1]]-0.5],
                vmin=0, vmax=vmax,fignum=False, cmap=cm_heat, origin="ij");
    
    [plt.plot(d/1000, which_ifreq, ".", color=dist2col(d)) for d in which_dists]
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().set_xticks(pdists/1000)
    plt.xticks(rotation=45, fontsize=8)    
    plt.xlabel("Distance (mm)")
    plt.ylabel("Frequency (Hz)", labelpad=-1)
    plt.title("Mismatch",pad=-2)
    plt.axis("auto")
    plt.colorbar()
    pdmax = pdists[np.argmin(np.abs(pdists - 100*1000))]
    plt.xlim((pdists[0]-dd/2)/1000,(pdmax+dd/2)/1000) #pdists[-1]/1000-0.5)

    plt.tight_layout(h_pad = 0, w_pad=0.2) #, w_pad = 0)
        