import os
import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
plt.style.use("default")
from matplotlib import colors as mcolors
from matplotlib.gridspec   import GridSpec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as mtrans
from scipy.stats import mannwhitneyu,ttest_1samp
import imageio as iio # For importing field snapshot PNGs
from collections import defaultdict
import fisher_plumes_fig_tools as fpft
import fisher_plumes_tools as fpt
from boulder import concs2rgb

import utils
import logging

from units import UNITS

logger = utils.create_logger("fisher_plumes_tools")
logger.setLevel(logging.DEBUG)

INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

pitch_sym = "ϕ"

scaled2col = lambda s, scale=1., cmap=cm.cool_r: cmap(s/scale)
dist2col   = lambda d, d_scale = 120000, cmap = cm.cool_r: scaled2col(d, d_scale, cmap)
freq2col   = lambda f, f_scale = 10,     cmap = cm.cool_r: scaled2col(f, f_scale, cmap)

def plot_two_plumes(F, which_idists, t_lim, which_probe = 0, dt = 0.5 * UNITS.sec, y_lim = None, y_ticks = None, axes = None, pos_dists = True, centered=True, cols=["r","b"]):
    to_sec  = lambda t: t.to(UNITS.sec).magnitude
    to_pitch= lambda x: x.to(UNITS(F.pitch_units)).magnitude
    d_scale = F.pitch.to(UNITS.um).magnitude
    dists  = np.array(sorted(F.rho[which_probe].keys()))       
    if pos_dists: dists  = dists[dists>0]
    ax_trace = []
    for i, di in enumerate(which_idists):
        ax_trace.append(plt.subplot(len(which_idists),1,i+1) if axes is None else axes[i])
        p = F.pairs_um[dists[di]]        
        if centered:
            balance = [np.exp(np.abs(np.log(np.abs(y1/y2)))) for (y1,y2) in p]
            ipair   = np.argmin(balance) # Find the pair where y1~y2
        else:
            ipair = 0
        ia, ib = p[ipair]
        a = F.sims[ia].data[:, which_probe].flatten()
        b = F.sims[ib].data[:, which_probe].flatten()
        t = F.sim0.t
        sc = max(a.std(), b.std())
        a /= sc
        b /= sc
        ax_trace[-1].plot(to_sec(t),a,color=cols[0], label=f"y={ia/d_scale:.2g} p", linewidth=1)
        ax_trace[-1].plot(to_sec(t),b,color=cols[1], label=f"y={ib/d_scale:.2g} p", linewidth=1)
        (i < 2) and ax_trace[-1].set_xticklabels([])
        (i ==2) and ax_trace[-1].set_xlabel("Time (sec.)", labelpad=-1)
        fpft.spines_off(ax_trace[-1])
        ax_trace[-1].set_xlim([to_sec(t_lim[0]), to_sec(t_lim[1])])
        ax_trace[-1].set_xticks(np.arange(to_sec(t_lim[0]), to_sec(t_lim[-1])+0.01,to_sec(dt)))
        y_lim is not None and ax_trace[-1].set_ylim(*y_lim)
        y_lim = ax_trace[-1].get_ylim()
        ax_trace[-1].set_yticks((np.arange(min(y_lim),max(y_lim)+1,5)) if y_ticks is None else y_ticks)
        ax_trace[-1].tick_params(axis='both', labelsize=8)
        ax_trace[-1].set_ylabel("Conc.", labelpad=-1)
        wndf = lambda x: x[(t>=t_lim[0])*(t<t_lim[-1])]
        aw, bw = wndf(a), wndf(b)
        ρ_w = np.corrcoef(aw,bw)[0,1]
        ρ   = np.corrcoef(a, b)[0,1]
        ax_trace[-1].set_title(f"$\Delta$ = {dists[di]/d_scale:.2g} {pitch_sym}, $\\rho_w$ = {ρ_w:.2f}, $\\rho$ = {ρ:.2g}", fontsize=8, verticalalignment="top")
        ax_trace[-1].xaxis.set_major_formatter(lambda x, pos: f"{x:g}")

    return ax_trace


def clip_snapshots(fields, lev=0.8):
    limsx, limsy = [], []
    for k, X in fields.items():
        my    = np.array([np.median(Xi) for Xi in X])
        limsy.append([np.where(my<lev)[0][0], np.where(my<lev)[0][-1]])
        mx    = np.array([np.median(Xi) for Xi in X.T])
        limsx.append([np.where(mx<lev)[0][0], np.where(mx<lev)[0][-1]])
    limsy = np.array(limsy).mean(axis=0).astype(int)
    limsx = np.array(limsx).mean(axis=0).astype(int)
    fclip = {k:X[limsy[0]:limsy[-1]][:, limsx[0]:limsx[-1]] for k, X in fields.items()}
    return fclip, limsx, limsy


def get_snapshot(fld, time, which_dim = 0, which_channel = 0, normalizer = 255., snapshots_dir = "."):
    file_name = os.path.join(snapshots_dir, f"{fld}_d{which_dim}_{int(time.to(UNITS.ms).magnitude):06}.png") 
    if not os.path.exists(file_name): raise FileExistsError(f"Could not find {file_name=}.")
    img = iio.imread(file_name)
    return np.array(img[:,:,which_channel])/normalizer    

def plot_plumes_demo(F, t_snapshot, 
                     which_keys,
                     which_probe = 0,
                     which_idists = [0,1,2],
                     t_wnd = [-4,4],
                     y_lim = (0,3),
                     mean_subtract_y_coords = True,
                     dt = 0.5,
                     figsize=(8,3),
                     t_center = None,
                     data_dir = None,
                     **kwargs
):
    to_pitch = lambda x: x.to(UNITS(F.pitch_string)).magnitude
    d_scale = F.pitch.to(UNITS.um).magnitude
    if "boulder" in F.pitch_string:
        fields = F.load_saved_snapshots(t = t_snapshot.to(UNITS.sec).magnitude, data_dir = data_dir)
    else:
        if hasattr(F, "sims"):
            fields_orig = {k:s.get_snapshot("S1", t_snapshot.to(UNITS.sec)) for k,s in F.sims.items()}
        else:
            print("No 'sims' attribute found. Trying to load snapshots from disk.")
            data_root = os.path.join(os.environ["FISHER_PLUMES_DATA"], "crick", F.name)
            snapshots_dir = lambda um: os.path.join(data_root, f"Y0.{int(um/1000)}", "png")
            fields_orig = {k:get_snapshot("S1", t_snapshot.to(UNITS.sec), snapshots_dir = snapshots_dir(k)) for k in F.yvals_um}
            
        print(list(fields_orig.keys()))
        fields, limsx, limsy = clip_snapshots(fields_orig)
        INFO(f"Clipped snapshots to {limsx=}, {limsy=}.")
    plt.figure(figsize=figsize)
    gs = GridSpec(3,3)
    ax_plume = plt.subplot(gs[:,0])
    pp = concs2rgb(fields[which_keys[0]], fields[which_keys[1]]) if fields else None
    dy = (F.sim0.y_lim[1] + F.sim0.y_lim[0])/2 if mean_subtract_y_coords else 0
    if pp is not None:
        ax_plume.matshow(pp, extent =
                         [to_pitch(x) for x in F.sim0.x_lim] +
                         [to_pitch(y - dy) for y in F.sim0.y_lim])
        px, py = [to_pitch(z) for z in F.sim0.get_used_probe_coords()[which_probe]]
        py -= to_pitch(dy)
        ax_plume.plot(px, py, "kx", markersize=5)
        ax_plume.xaxis.set_ticks_position('bottom')
#        ax_plume.axis("equal")
    plt.xlabel(f"x ({pitch_sym})", labelpad=-1)
    plt.ylabel(f"y ({pitch_sym})", labelpad=-1)
    #ax_plume.set_yticks(arange(-0.2,0.21,0.1) if 'wide' in name else arange(-0.1,0.11,0.1))

    if t_center is None: t_center = t_snapshot
    ax_trace = plot_two_plumes(F, which_idists, t_lim  = t_wnd + t_center,
                               dt = dt, y_lim = y_lim,
                               axes = [plt.subplot(gs[i,1]) for i,_ in enumerate(which_idists)], **kwargs)
        
    ax_corr_dist = plt.subplot(gs[:,-1])
    rho   = F.rho[which_probe]
    dists = np.array(sorted(list(rho.keys()))) 
    rho   = {d:rho[d][0] for d in dists} # Take the raw data, not the bootstraps
    rhom  = np.array([np.mean(np.sum(rho[d],axis=0)) for d in dists])
    rhos  = np.array([ np.std(np.sum(rho[d],axis=0)) for d in dists])
    col   = "gray"
    plt.fill_between(dists/d_scale, rhom-rhos,rhom+rhos, color=fpft.set_alpha(mpl.colors.to_rgba(col),0.2));
    fpft.pplot(dists/d_scale, rhom , "o-", markersize=4,color=col);
    ax_corr_dist.grid(True, linestyle=":")
    #ax_corr_dist.set_yticklabels(["-1","","0","","1"])
    ax_corr_dist.set_ylabel("Correlation",labelpad=-1)
    ax_corr_dist.set_xlabel(f"Intersource distance ({pitch_sym})", labelpad=-1)
    ax_corr_dist.xaxis.set_major_formatter(lambda x, pos: f"{x:g}")
    plt.tight_layout(pad=0,w_pad=0,h_pad=1)

    return ax_plume, ax_trace, ax_corr_dist


def plot_correlations(rho,
                      d_scale,
                      slices = {"all":slice(1,10000), "first":slice(1,2),   "second":slice(10,11)}, # Which frequency slices to plot
                      cols   = {"all":cm.gray(0.4),   "first":cm.cool(1.0), "second":cm.cool(0.2)},
                      n_rows = 1,
                      plot_order = None, 
                      figsize=None,
                      plot_slices = True,
                      plot_overlay = True,
                      ax = [],
                      legend_args = {},
):

    
    dists    = np.array(sorted(list(rho.keys())))
    dists_p  = dists/d_scale    
    rho      = {d:rho[d][0] for d in dists} # [0] to take the raw data
    rho_mean = {k:np.array([np.mean(np.sum(rho[d][slc,:],axis=0)) for d in dists]) for k, slc in slices.items()}
    rho_std  = {k:np.array([ np.std(np.sum(rho[d][slc,:],axis=0)) for d in dists]) for k, slc in slices.items()}
    rho_pc   = {k:{pc:np.array([np.percentile(np.sum(rho[d][slc,:],axis=0), pc) for d in dists]) for pc in [5,50,95]} for k, slc in slices.items()}

    n_slices = len(slices)
    n_panels = n_slices + 1
    n_cols   = int(np.ceil(n_panels/n_rows))

    gave_axes = len(ax) > 0
    
    not gave_axes and plt.figure(figsize=figsize if figsize else (3*n_cols, 3 * n_rows))
    if not gave_axes: ax = []

    if plot_slices:
        gs = GridSpec(n_rows, n_cols)
        if plot_order is None: plot_order = list(slices.keys())
        for i, k  in enumerate(plot_order):
            irow = int(i // n_cols)
            icol = int(i %  n_cols)
            if not gave_axes: ax.append(plt.subplot(gs[irow, icol]))
            else: plt.sca(ax[i])
            sc= 1
            y = rho_mean[k]/sc
            ee= rho_std[k]/sc
            plt.fill_between(dists_p, y-ee, y+ee, color=fpft.set_alpha(mpl.colors.to_rgba(cols[k]),0.2));
            fpft.pplot(dists_p, y , "o-", markersize=4,color=cols[k]);
            plt.xlabel("Distance (p)")
            (icol == 0) and plt.ylabel("Correlation")
            plt.title(k)

    if plot_overlay:
        if not gave_axes: ax.append(plt.subplot(gs[int(n_slices//n_cols), int(n_slices % n_cols)]))
        else: plt.sca(ax[-1])
        
        for i, k in enumerate(slices):
            fpft.pplot(dists_p, rho_mean[k]/max(rho_mean[k]), "-", markersize=4, color=cols[k], label=k);
        plt.legend(frameon=False, fontsize=8, **legend_args)
        plt.xlabel("Distance (p)")        
        plt.title("Overlayed and Scaled")

    if plot_slices or plot_overlay:
        plt.tight_layout(w_pad=0)

    return ax

def plot_scattergram(coefs, ifreq, i_pos_dist, pairs_um, pitch_units,
                  figsize=(8,3),
                     dist_col_scale = 120000,
                     markersize=10,
                     cols = ["C0","C1","C2","C3"],
                     coef_names = {0:"Coef 1", 1:"Coef 2"},
                     plot_grid = False,
                     lim_scale = 4,
):
    
    if type(coefs) is not list: coefs = [coefs]

    dists   = np.array(sorted(pairs_um.keys()))
    which_d = dists[dists>0][i_pos_dist]
    pooled1  = np.concatenate([
        [coef[y][0][:, ifreq] for y in (y1,y2) for coef in coefs] for (y1,y2) in pairs_um[which_d]],
                             axis = 1)

    plt.figure(figsize=figsize)
    th = np.linspace(0,2*np.pi,1001)

    dist2col_ = lambda d: dist2col(d,dist_col_scale)

    axes = []
    gs = GridSpec(len(pooled1), len(pooled1))
    for i, p0 in enumerate(pooled1):
        for j, p1 in enumerate(pooled1):
            axes.append(plt.subplot(gs[i,j]))
            U,S,_ = np.linalg.svd(np.cov([p0,p1]))
            p0m= np.mean(p0)
            p1m= np.mean(p1)
            plt.plot(p0,  p1, "o", color=fpft.mix_colors([cols[i],cols[j]]), markersize=markersize)
            plt.plot(p0m, p1m, "k+", markersize=10)
        
            for r in [1,2,3]:
                xy = U @ np.diag(np.sqrt(S)) @ [np.cos(th), np.sin(th)]
                plt.plot(r*xy[0] + p0m, r*xy[1]+p1m,":",color="k", linewidth=1)

            pc = np.percentile(np.concatenate([p0,p1]).flatten(),[1,99])
            scale = lim_scale*max(abs(pc))
            plt.xlim([-scale,scale]);
            plt.ylim([-scale,scale])
            fpft.spines_off(plt.gca())
            #plt.axis("equal")
            plt.grid(True, linestyle=":", color="k", alpha=0.5)
            if i==len(pooled1)-1: plt.xlabel(f"{coef_names[j%2]}, Src {(j//2)+1}")
            if j==0: plt.ylabel(f"{coef_names[i%2]}, Src {(i//2)+1}")
    return axes

                  

def plot_coef1_vs_coef2(coefs, ifreq, pairs_um, pitch_units,
                        figsize=(8,3),
                        i_pos_dists_to_plot = [0,2,4],
                        dist_col_scale = 120000, axes = None,
                        
):
    if type(coefs) is not list: coefs = [coefs]
        
    pooled1 = {d:np.array([
        np.concatenate([coef[y1][0][:, ifreq] for coef in coefs for (y1,y2) in pairs_um[d]], axis = 0),
        np.concatenate([coef[y2][0][:, ifreq] for coef in coefs for (y1,y2) in pairs_um[d]], axis = 0)
    ]) for d in pairs_um} # [0] takes the raw data, not the bootstraps

    dists = np.array(sorted(pooled1.keys()))
    plt.figure(figsize=figsize)
    th = np.linspace(0,2*np.pi,1001)
    which_d = [dists[dists>0][i] for i in i_pos_dists_to_plot]

    dist2col_ = lambda d: dist2col(d,dist_col_scale)
    if (axes is not None) and len(axes) != len(which_d):
        raise ValueError("Number of axes provided {len(axes)} != number of distances {len(which_d)}.")
    axes = [] if axes is None else axes
    for i, d in enumerate(which_d):
        axes.append(plt.subplot(1,len(which_d),i+1)) if len(axes)<len(which_d) else plt.sca(axes[i])
        U,S,_ = np.linalg.svd(np.cov(pooled1[d]))
        p0 = pooled1[d][0]
        p1 = pooled1[d][1]
        p0m= np.mean(p0)
        p1m= np.mean(p1)
        plt.plot(p0,p1, "o", color=dist2col_(d), markersize=1)
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
        d *= UNITS.um
        plt.title(f"{d.to(UNITS(pitch_units)).magnitude:.2g} {pitch_sym}")
        plt.grid(True)
    return axes

def plot_coef_vs_coef_and_traces(F, freq, idists_to_plot, which_probe = 0,
                                 dt = 0.5 * UNITS.sec, t_lim = [19,21]*UNITS.sec,
                                 y_lim = None, n_per_row=1, rel_trace_width=2, **kwargs):
    n_dists = len(idists_to_plot)
    n_rows  = int(np.ceil(n_dists/n_per_row))
    gs      = GridSpec(n_rows, (1 + rel_trace_width)*n_per_row)
    
    gs_trace_fun = lambda i: gs[i//n_per_row, 1 + (1 + rel_trace_width)*(i % n_per_row):1+rel_trace_width + (1 + rel_trace_width)*(i % n_per_row)]
    gs_coef_fun  = lambda i: gs[i//n_per_row, (1 + rel_trace_width)*(i % n_per_row)]
    trace_axes   = [plt.subplot(gs_trace_fun(i)) for i,_ in enumerate(idists_to_plot)]
    coef_axes    = [plt.subplot(gs_coef_fun(i))  for i,_ in enumerate(idists_to_plot)]
    
    plot_coef1_vs_coef2([F.ss[which_probe], F.cc[which_probe]], F.freqs2inds([freq])[0], F.pairs_um, F.pitch_string, i_pos_dists_to_plot = idists_to_plot, axes = coef_axes, **kwargs)
    plot_two_plumes(F, idists_to_plot, t_lim, which_probe = which_probe, dt = dt, y_lim = y_lim, axes = trace_axes)
    return coef_axes, trace_axes
    
def plot_alaplace_fits(F, which_dists_um,
                       which_probe = 0,
                       ifreq_lim = [], which_ifreq = 1, 
                       figsize=None, vmin=None,vmax=None,
                       heatmap_xmax = np.inf,
                       heatmap_default_xticks = False,
                       plot_dvals = False,
                       plot_pvals = False):
    if figsize is not None:
        plt.figure(figsize=figsize)

    d_scale = F.pitch
    
    gs   = GridSpec(2 if plot_dvals else 1, len(which_dists_um)+2)
    dmax = -1
    yld  = []
    ax_dcdf= []    
    ax_cdf = []
    for di, d in enumerate(which_dists_um):
        INFO(f"{d=:3d} @ Freq # {which_ifreq:3d}: -np.log10(p) = {-np.log10(F.pvals[which_probe][d][0][which_ifreq]):1.3f}")
        ax_cdf.append(plt.subplot(gs[:-1 if plot_dvals else 1,di]))
        rr    = F.rho[which_probe][d][0][which_ifreq]
        xl    = fpft.expand_lims([np.min(rr), np.max(rr)],1.2)        
        xvals = np.linspace(xl[0],xl[-1],1001)

        la_   = F.la[which_probe][d][0][which_ifreq]
        mu_   = F.mu[which_probe][d][0][which_ifreq]
        ypred = fpt.alaplace_cdf(la_, mu_, xvals)

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
        plt.title(f"{(d * UNITS.um).to(UNITS(F.pitch_string)).magnitude:.2g} {pitch_sym}")

        ax_cdf[-1].xaxis.set_major_formatter(lambda x, pos: f"{x:g}")
        ax_cdf[-1].yaxis.set_major_formatter(lambda x, pos: f"{x:g}")        

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
            ax_dcdf.append(axdi)
            plt.grid(True, axis='y', linestyle=":")
            #(di != 0 ) and axdi.set_yticklabels([])
            plt.xlabel("x",labelpad=-2)
            (di == 0) and plt.ylabel("|F$_{data}$($x$) - F$_{fit}$($x$)|")
            axdi.xaxis.set_major_formatter(lambda x, pos: f"{x:g}")
            axdi.yaxis.set_major_formatter(lambda x, pos: f"{x:g}")                    

    if plot_dvals:
        [axdi.set_ylim(yld) for axdi in ax_dcdf]

    freq_res = F.fs/F.wnd
    ax_hm = []
    for i,vals in enumerate([F.pvals[which_probe], F.r2vals[which_probe]]):
        if not plot_pvals and i == 0: continue

        ax_hm.append(plt.subplot(gs[i,-2:] if plot_pvals else gs[:,-2:]))

        dists_um = np.array(sorted(vals))
        n_dists = len(dists_um)
        dd = np.mean(np.diff(dists_um))
        v = np.array([vals[d][0] for d in sorted(vals)]).T
        if len(ifreq_lim)==0:
            ifreq_lim = [0, v.shape[0]]
        v = v[ifreq_lim[0]:ifreq_lim[1],:]
        
        # The distances are non-uniform, so just have one tick each and label them according to the actual distance
        extent = [-0.5, n_dists-0.5, (F.freqs[ifreq_lim[0]]-freq_res/2).magnitude, (F.freqs[ifreq_lim[1]]-freq_res/2).magnitude]
        print(f"Setting extent to {extent}.")
        plt.matshow(-np.log10(v) if i==0 else v, #+np.min(p[p>0])/10),
                extent = extent,
                    vmin=0 if vmin is None else vmin[i], vmax=None if vmax is None else vmax[i],fignum=False,
                    cmap=cm.RdYlBu_r if i == 0 else cm.RdYlBu,origin="lower");
        
        [plt.plot(list(dists_um).index(d), F.freqs[which_ifreq], ".", color=dist2col(d)) for d in which_dists_um]
        plt.xticks(np.arange(n_dists), labels=[f"{(di * UNITS.um).to(UNITS(F.pitch_string)).magnitude:.2g}" for di in dists_um],
                   fontsize=6, rotation=90)
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.xlabel(f"Distance ({pitch_sym})",labelpad=0)
        #(i == 0) and plt.gca().set_xticklabels([])
        plt.ylabel("Frequency (Hz)", labelpad=-1)
        plt.title("Mismatch (p-value)" if i==0 else "Match ($R^2$)",pad=-2)
        plt.axis("auto")
        plt.colorbar()
    
    plt.tight_layout(h_pad = 0, w_pad=0.2) #, w_pad = 0)

    return ax_cdf, ax_dcdf, ax_hm
        
def plot_gen_exp_parameter_fits_panel(F, which_fis, contours_dist = None,
                                      which_probe=0,
                                      n_contours = 12, contours_cmap=cm.gray,
                                      plot_scatter = True,
                                      plot_legend = True,
                                      log_scale = False,
                                      label_color = "black",
                                      plot_others = False,
                                      xt = None,
                                      yt = None,
                                      scatter_alpha= 0.8, scatter_size=3,
                                      colfun = lambda f: freq2col(f, 10, cmap=cm.cool_r)
):
    INFO(f"plot_gen_exp_paramter_fits_panel with {which_fis=}, {log_scale=}.")
    d_scale = F.pitch.to(UNITS.um).magnitude
    fun = np.log10 if log_scale else (lambda X: X)
    γbs = fun(F.fit_params[which_probe][1:][:, which_fis, 1]/d_scale) # 1: is to take the bs runs.
    kbs = fun(F.fit_params[which_probe][1:][:, which_fis, 2])

    if plot_scatter:
        # If the standard deviation is very small, then assume we've fixed k,
        # and plot box and whisker plots instead, using the data in the columns
        # of γbs.
        if np.std(kbs[0]) < 1e-5:
            # Set the color of the boxplots to the color of the frequency
            # (which is the same as the color of the scatter points).
            
            box = plt.boxplot(γbs, widths = 0.3, patch_artist=True)
            [plt.setp(b, color = fpft.set_alpha(colfun(f),0.5), facecolor=fpft.set_alpha(colfun(f),0.25)) for f,b in zip(F.freqs[which_fis], box["boxes"])]
            [plt.setp(b, color = colfun(f), linewidth=1) for f, b in zip(F.freqs[which_fis], box["medians"])]

            plt.xticks(np.arange(1, len(which_fis)+1),
                       labels = [f"{F.freqs[i].to('Hz').magnitude:g} Hz" for i in which_fis])

            yl = plt.ylim()
            yt = plt.yticks()[0];
            log_scale and plt.yticks(yt, labels=[f"{10**yti:.2g}" for yti in yt]) and plt.ylim(yl)
            
            return 1 # number of parameters plotted
        else:
            for γ, k in zip(γbs, kbs):
                plt.scatter(γ, k,
                            c=[fpft.set_alpha(colfun(f), scatter_alpha) for i, f in enumerate(F.freqs[which_fis])],
                            s = scatter_size)
    
    if plot_others:
        ind_max = F.freqs2inds([F.freq_max])[0]
        other_inds = [fi for fi in range(1, ind_max+1) if fi not in which_fis]
        cols = [fpft.set_alpha(cm.Oranges((fi/F.wnd)*F.fs/F.freq_max),0.5) for fi in other_inds]
        γm = np.mean(fun(F.fit_params[which_probe][1:][:, other_inds, 1]/d_scale),axis=0)
        κm = np.mean(fun(F.fit_params[which_probe][1:][:, other_inds, 2]),axis=0)
        plt.scatter(γm, κm, c = cols, s=3, marker="o", edgecolors=None, linewidth=1, zorder=2)
        

    hmus = []
    for ifreq, fi in enumerate(which_fis):
        γk = np.array([np.array(γbs[:,ifreq]), np.array(kbs[:,ifreq])])
        mu = np.mean(γk,axis=1)
        C  = np.cov(γk)
        hmui, hsdsi = fpft.plot_bivariate_gaussian(mu, C,
                                                   n_sds = 1, n_points = 1000,
                                                   mean_style = {"marker":"o", "markersize":3, "color":colfun(F.freqs[fi])},
                                                   sd_style   = {"linewidth":1, "color":colfun(F.freqs[fi])},
        )
        hmus.append(hmui)

        plot_legend and plt.legend(handles = hmus, labels=[f"{F.freqs[fi].magnitude:g} Hz" for fi in which_fis],
                                   labelcolor = label_color,
                                   labelspacing=0,
                                   frameon=False,
                                   fontsize=6,
    )
    xl = list(plt.xlim())
    if xt is not None:
        xl[0] = min(xl[0], np.min(xt))
        xl[1] = max(xl[1], np.max(xt))
        
    yl = list(plt.ylim())
    if yt is not None:
        yl[0] = min(yl[0], np.min(yt))
        yl[1] = max(yl[1], np.max(yt))

    if n_contours:
        contours_dist_p = np.array(contours_dist)/d_scale
        γγ, kk = np.meshgrid(np.linspace(*xl, 101), np.linspace(*yl,101))
        if log_scale:
            I = 2*kk - (10**kk)*γγ + (10**kk - 2) * np.log10(contours_dist_p)
        else:
            I = 2*np.log10(kk) - kk*np.log10(γγ) + (kk - 2) * np.log10(contours_dist_p)
        plt.contourf(γγ, kk, I, n_contours, cmap=contours_cmap)  

    if xt is None:
        xl = plt.xlim()
        xt = plt.xticks()[0];
        log_scale and plt.xticks(xt, labels=[f"{10**xti:.2g}" for xti in xt]) and plt.xlim(xl)
    else:
        plt.xticks(xt, labels=[f"{(10**xti) if log_scale else xti:g}" for xti in xt])
    
    if yt is None:
        yl = plt.ylim()
        yt = plt.yticks()[0];
        log_scale and plt.yticks(yt, labels=[f"{10**yti:.2g}" for yti in yt]) and plt.ylim(yl)
    else:
        plt.yticks(yt, labels=[f"{(10**yti) if log_scale else yti:g}" for yti in yt])

        
    fpft.spines_off(plt.gca())

    plt.ylabel("Exponent $k_n$")
    plt.xlabel(f"Length Scale $\gamma_n$ ({pitch_sym})")
    plt.title("Fit Parameters")

    return 2
    
def plot_la_gen_fits_vs_distance(F,
                                 which_probe = 0,
                                 which_ifreqs = [1,2,3,4],
                                 figsize = None, legloc = None, xl = None,
                                 colfun = lambda f: freq2col(f, 10),
                                 **kwargs
):

    (figsize is not None) and plt.figure(figsize=figsize)

    gs     = GridSpec(2,3)
    dd_all_um = np.array(sorted(list(F.la[which_probe].keys())))
    la     = {d:F.la[which_probe][d][0] for d in F.la[which_probe]}
    la_sub = np.array(la[d] for d in dd_all_um if d in F.dd_fit)
    freqs  = np.arange(F.wnd)/F.wnd*F.fs
    ax       = []
    max_norm = lambda y: y/np.max(y)
    identity = lambda y: y

    d_scale = F.pitch.to(UNITS.um).magnitude
    dx = dd_all_um / d_scale
    for i, fi in enumerate(which_ifreqs):
        row   = i // 2
        col   = i %  2
        ax.append(plt.subplot(gs[row, col]))
        X = np.stack([F.la[which_probe][d][1:, fi] for d in dd_all_um],axis=1) # 1: is to take the bootstraps (0 is the raw data)
        la_lo, la_med, la_hi = np.percentile(X, [5,50,95], axis=0)
        ax[-1].plot(dx, la_med, "o-", color=colfun(F.freqs[fi]), linewidth=1, markersize=2, label=f"{freqs[fi].magnitude:g} Hz")
        ax[-1].plot([dx,dx], [la_lo, la_hi], "-", color=fpft.set_alpha(colfun(F.freqs[fi]),0.5), linewidth=1)
        for j in range(min(5, F.n_bootstraps)):
            ax[-1].plot(F.dd_fit/d_scale, fpt.gen_exp(F.dd_fit, *(F.fit_params[which_probe][1+j][fi])),
                        color="lightgray", #fpft.set_alpha(colfun(fi),0.5),
                        linewidth=1, zorder=-5)
        (row == 1) and plt.xlabel(f"Distance $s$ {pitch_sym}")
        (col == 0) and plt.ylabel("$\lambda_n(s)$")        
        (xl is not None) and plt.xlim(xl)
        plt.title(f"{freqs[fi].magnitude:g} Hz", pad=-2)
        fpft.spines_off(plt.gca())

    # PLOT THE PARAMETERS
    ax.append(plt.subplot(gs[:,-1]))
    # n_params is 2 if two paramters were plotted, or 1 if only one was plotted (e.g. when k is fixed)
    n_params = plot_gen_exp_parameter_fits_panel(F, which_ifreqs, which_probe = which_probe, colfun = colfun, n_contours = 0, **kwargs)    
    fpft.spines_off(plt.gca())
    if n_params == 2:
        plt.ylabel("Exponent $k_n$")
        plt.xlabel(f"Length Scale $\gamma_n$ ({pitch_sym})")
        plt.title("Fit Parameters")
    elif n_params == 1:
        plt.ylabel(f"Length Scale $\gamma_n$ ({pitch_sym})")
        plt.xlabel(f"Frequency")
        plt.title("Fit Parameters")
        
    return ax
    
def plot_fisher_information(
        F,
        which_probe=0,
        d_lim_um =[1e-3,100],
        d_vals_um = [0.1,1,10],
        d_space_fun = np.linspace,
        which_ifreqs = [2,4,6,8,10],
        x_stagger = lambda x, i: x*(1.02**i),
        plot_fun = plt.plot,
        freq_max = None,
        bf_ytick = None,
        colfun = lambda f: cm.cool_r(f/10.),
        plot_ests = False,
        plot_param_fits = False,
        info_heatmap = True,
        heatmap_range = [None,None],
        heatmap_cm    = cm.plasma,
        **kwargs
):
    d_scale = F.pitch.to(UNITS.um).magnitude
    d0,d1 = d_lim_um
    d1 = max(d1, np.max(F.I_dists[which_probe])) # So that the line plot spans at least as far as the medians+/-whiskers
    dd = d_space_fun(d0, d1,101)
    INFO(f"{dd[0]=:g}, {dd[-1]=:g} ({dd[0]/d_scale:g}, {dd[-1]/d_scale:g} ϕ)")
    Idd= F.compute_fisher_information_at_distances(dd)  # bs * freq * dists
    Idd_med = np.median(Idd[which_probe][1:],axis=0) * d_scale**2 # Median over bootsraps

    Ilow_est_dd, Ihigh_est_dd = zip(*F.compute_fisher_information_estimates_at_distances(dd))
    Ilow_est_dd_med  = np.median(Ilow_est_dd[which_probe][1:],axis=0) * d_scale**2
    Ihigh_est_dd_med = np.median(Ihigh_est_dd[which_probe][1:],axis=0) * d_scale**2     
    
    # which_ifreqs = list(set(F.I_best_ifreqs_mode[which_probe]))
    
    colfun = lambda fi: cm.cool_r(list(sorted(which_ifreqs)).index(int(fi))/len(which_ifreqs))    
    colfun = lambda f: cm.cool_r(f.to(UNITS.Hz).magnitude/(freq_max if freq_max is not None else F.freq_max).to(UNITS.Hz).magnitude)
    freqs = F.freqs
    
    INFO(f"Plotting {which_ifreqs=}.")
    n_dvals = len(d_vals_um)
    gs      = GridSpec(6,n_dvals)

    ax_fisher = plt.subplot(gs[:3,:])
    Ilow, Imed, Ihigh = np.array([F.I_pcs[which_probe][pc] for pc in sorted(F.I_pcs[which_probe])]) * d_scale**2

    colfuni = lambda i: cm.cool_r(i/4) #f"C{i}"
    for i, (fi, Il, Im, Ih) in enumerate(zip(which_ifreqs, Ilow[which_ifreqs], Imed[which_ifreqs], Ihigh[which_ifreqs])):
        x = x_stagger(dd/d_scale, i)
        col = colfuni(i) # colfun(freqs[fi])
        plot_fun(x, Idd_med[fi], "-", linewidth=1,markersize=2,
                 color=col,
                 label = f"{F.freqs[fi].magnitude:g} Hz")
        if plot_ests:
            plot_fun(x, Ilow_est_dd_med[fi],  ":", linewidth=1,markersize=2,color=col, label = f"{F.freqs[fi].magnitude:g} Hz (low s)")
            plot_fun(x, Ihigh_est_dd_med[fi], "--", linewidth=1,markersize=2,color=col, label = f"{F.freqs[fi].magnitude:g} Hz (high s)")                
        x = x_stagger(F.I_dists/d_scale, i)
        plot_fun(x, Im, "o", linewidth=1,markersize=2,color=col)        
        plot_fun([x, x], [Il, Ih], color = col, linewidth=0.5)

    plt.legend(frameon=False, labelspacing=0.25,fontsize=8)
    plt.ylabel(f"Fisher Information ({pitch_sym}" + "$^{-2}$)")
    ax_fisher.tick_params(axis='x', labelsize=8)
    [lab.set_y(0.01) for lab in ax_fisher.xaxis.get_majorticklabels()]
    #ax_fisher.set_xlim(np.floor(d0/d_scale), np.ceil(d1/d_scale))
    if plot_param_fits:
        #ax_fisher.text(0.4,0.025,f"Distance ({pitch_sym})", fontsize=11, transform=ax_fisher.transAxes)
        ax_fisher.set_xlabel(f"Distance ({pitch_sym})", fontsize=11, labelpad=-30)
    else:
        ax_fisher.set_xlabel(f"Distance ({pitch_sym})")
    fpft.spines_off(plt.gca())

    # The best frequency plots
    ax_best_freq = plt.subplot(gs[3,:] if plot_param_fits else gs[3:,:])
    if info_heatmap:
        plot_fisher_information_heatmap(F, which_probe,
                                        ax = ax_best_freq,
                                        freq_max = freq_max,
                                        heatmap_range = heatmap_range,
                                        heatmap_cm    = heatmap_cm,
                                        do_colorbar = True)

        if plot_param_fits:
            ax_best_freq.tick_params(axis='y', labelsize=8)
            ax_best_freq.set_ylabel("Freq. (Hz)", labelpad=-0.5)
            ax_best_freq.set_xticks(np.arange(len(dp))+0.5, labels=[f"{z:.2g}" for z in dp],
                                    fontsize=8,
                                    rotation=90,
                                    color="black")                
            ax_best_freq.tick_params(top=False, labeltop=False)
            # Label the bottom xticks
            ax_best_freq.tick_params(bottom=False, labelbottom=True)
            for i, (di, label) in enumerate(zip(dp, ax_best_freq.get_xticklabels())):
                label.set_color("k" if di<0.6 else "w")
                label.set_transform(label.get_transform() + mtrans.Affine2D().translate(0, 70))
            ax_best_freq.xaxis.set_label_position('top')            
            ax_best_freq.set_xlabel(f"Distance ({pitch_sym})", fontsize=11,labelpad=-12, color="w")
        
    else:
        pc = np.percentile(F.I_best_freqs[which_probe], [5,50,95], axis=0)    
        ax_best_freq.semilogx(F.I_dists/d_scale,
                              pc[1],
                              "-",color="lightgray",markersize=3, linewidth=1,zorder=-5)
        cols = [colfun(freqs[fi]) for fi in F.I_best_ifreqs_mode[which_probe]]
        cols = [cm.gray(0.4) for fi in F.I_best_ifreqs_mode[which_probe]]
        m = np.mean(F.I_best_freqs[which_probe],axis=0)
        ax_best_freq.scatter(F.I_dists/d_scale,
                             pc[1],
                             c=cols,
                             s=8)
        ax_best_freq.set_prop_cycle('color', [fpft.set_alpha(c,0.5) for c in cols])
        ax_best_freq.plot([F.I_dists/d_scale]*2, [pc[0],pc[-1]], linewidth=0.5)
        
        plot_param_fits and ax_best_freq.xaxis.tick_top()
        not plot_param_fits and ax_best_freq.set_xlabel(f"Distance ({pitch_sym})")    
        plt.ylabel("Frequency (Hz)", labelpad=-1)
        plot_param_fits and ax_best_freq.set_xticklabels([])
        ax_best_freq.set_xlim(ax_fisher.get_xlim())
        if bf_ytick is not None: ax_best_freq.set_yticks(bf_ytick)        
        fpft.spines_off(ax_best_freq, ["bottom" if plot_param_fits else "top", "right"])
    
    # The parameter fits plots
    ax_d = []
    if plot_param_fits:
        for i, d in enumerate(d_vals_um):
            ax = plt.subplot(gs[4:,i])
            plot_gen_exp_parameter_fits_panel(F, sorted(which_ifreqs), contours_dist = d,
                                              n_contours = 12, contours_cmap=cm.gray,
                                              plot_legend = (i==0),
                                              plot_scatter = False,
                                              plot_others = False,
                                              label_color = "white",
                                              #colfun = lambda f: colfun(freqs[which_ifreqs[list(F.freqs[which_ifreqs]).index(f)]]),
                                              colfun = colfun,
                                              
                                              **kwargs)
            ax.set_title(f"{d/d_scale:.2g} {pitch_sym}")
            (i != 0) and (ax.set_ylabel(""), ax.set_yticklabels([]))
            ax_d.append(ax)
                
    return ax_fisher, ax_best_freq, ax_d


def plot_fisher_information_heatmap(F, which_probe,
                                    ax = None,
                                    freq_max = None,
                                    heatmap_range = [None, None],
                                    heatmap_cm    = cm.Spectral_r,
                                    do_colorbar = True,
                                    
):
    d_scale = F.pitch.to(UNITS.um).magnitude    
    I = F.I[which_probe][0] * d_scale**2
    n_freqs, n_d = I.shape
    fI = F.freqs[:n_freqs]
    ind_use = fI > 0 * UNITS.Hz
    if freq_max is not None: ind_use &= fI <= freq_max
    used_freqs_hz = fI[ind_use].to(UNITS.Hz).magnitude
    if ax is None: ax = plt.gca()
    im = ax.matshow(np.log10(I[ind_use]),
                    origin='upper',
                    cmap=heatmap_cm,
                    aspect="auto",
                    vmin = heatmap_range[0],
                    vmax = heatmap_range[1],
                    extent = [0, n_d, used_freqs_hz[-1], used_freqs_hz[0]]
    )

    dp = F.I_dists/F.pitch.to(UNITS.um).magnitude
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel("Freq. (Hz)", labelpad=5)
    ax.set_xticks(np.arange(len(dp))+0.5, labels=[f"{z:.2g}" for z in dp],
                            fontsize=8,
                            rotation=90,
                            color="black",
    )

    ax.tick_params(top=False, labeltop=False)
    ax.tick_params(bottom=True, labelbottom=True)
    ax.xaxis.set_label_position('bottom')            
    ax.set_xlabel(f"Distance ({pitch_sym})")

    if do_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cb = plt.colorbar(im, cax=cax)
        yt = cb.ax.get_yticks()
        yl = cb.ax.get_ylim()
        labels = []
        for lab in [f"{10**yti:.1g}" for yti in yt]:
            if "e+" in lab:
                head, tail = lab.split("e+")
                labels.append(f"{int(head) * 10**int(tail)}")
            else:
                labels.append(lab)

        cb.ax.set_yticks(yt, labels)
        cb.ax.set_ylabel(f"Fisher Information ({pitch_sym}" + "$^{-2}$)")
        cb.ax.set_ylim(yl)
    else:
        cb = None

    return ax, cb

def plot_window_series(proc_data, figsize=(8,5), n_rows = 2, heatmap_cm = cm.Spectral_r, freq_max = None, heatmap_range = [-2, np.log10(500)], **kwargs):
    plt.figure(figsize=figsize)
    n_data = len(proc_data)
    order = sorted(proc_data.keys())
    gs = GridSpec(n_rows, int(np.ceil(len(order)/n_rows)))    
    axes, cbs = [], []
    for k, gsi in zip(order, gs):
        if freq_max is None: freq_max = proc_data[k].freq_max        
        axes.append(plt.subplot(gsi))
        axes[-1], cbi = plot_fisher_information_heatmap(proc_data[k], 0, ax = axes[-1], freq_max = freq_max,
                                                            heatmap_range =heatmap_range,
                                                            heatmap_cm    =heatmap_cm,
                                                            do_colorbar   = gsi.is_last_col(),
        )
        axes[-1].set_title(f"{k}")
        if not gsi.is_first_col(): axes[-1].set_ylabel("")
    plt.tight_layout()


def plot_length_constants_vs_frequency(data, which_ds, which_probe,
                                       names  = {"s=p":"Surrogate data", "bw":"Simulations", "16Ts":"Supplementary"},
                                       labels = {"s=p":"Surr", "bw":"Sims", "16Ts":"Supp"},
                                       cols   = {"s=p":cm.gray(0.4), "bw":cm.GnBu(0.75), "16Ts":cm.GnBu(0.35), },
                                       gamma_plot_width = 2,
                                       figsize = None):
                                       
    gs = GridSpec(len(which_ds), 1+gamma_plot_width+1)
    plt.figure(figsize=(8,2 * len(which_ds) if figsize is None else figsize))
    ax = []
    which_corr_freqs = [2,5,10,20] * UNITS.Hz
    labs = [f"{f}" for f in which_corr_freqs]
    cols.update({l:col for l,col in zip(labs, [cm.cool(1 - f.magnitude/20) for f in which_corr_freqs])})
    coef_γ_vs_freq = {}
    for i, ds in enumerate(which_ds):        
        F = data[ds]
        freq_min = F.fs/F.wnd #(1/window_length.to(UNITS.s).magnitude) * UNITS.Hz        
        coef_γ_vs_freq[ds] = F.coef_γ_vs_freq[which_probe]
        d_scale   = F.pitch.to(UNITS.um).magnitude
        freq_inds = F.freqs2inds(which_corr_freqs)
        slices    = {}    
        slices.update({l:slice(fi, fi+1) for l, fi in zip(labs, freq_inds)})    
        ax.append(plt.subplot(gs[i,0]))
        plot_correlations(F.rho[0], F.pitch.to(UNITS.um).magnitude, slices = slices, cols = cols,
                          plot_slices = False, plot_overlay=True, ax = [ax[-1]],
                          legend_args = {"handlelength":1.25, "handletextpad":0.5, "borderpad":0},
        )
        ax[-1].set_xlabel(f"Intersource distance ({pitch_sym})")
        ax[-1].set_title("")
        ax[-1].set_yticks(np.arange(0,1.1,0.5))
        ax[-1].set_ylabel("Correlations")

        ax.append(plt.subplot(gs[i,1:1+gamma_plot_width], sharey=None))# if not i else ax[1]))
        γbs = F.fit_params[0][1:][:,:,1]
        freq_max = F.freq_max
        freqs_γ  = F.freqs[:γbs.shape[1]]
        ind_use  = (freqs_γ > freq_min) & (freqs_γ <= freq_max)
        lo,md,hi = np.percentile(γbs, [5,50,95], axis = 0) / d_scale
        
        ax[-1].plot(freqs_γ[ind_use].to(UNITS.Hz).magnitude, md[ind_use], "o",color=cols[ds], markersize=4)
        ax[-1].plot([freqs_γ[ind_use].to(UNITS.Hz).magnitude]*2, [lo[ind_use], hi[ind_use]], color=cols[ds],linewidth=1)
        ax[-1].set_xlabel("Frequency (Hz)")
        ax[-1].set_ylabel(f"Length constant ({pitch_sym})")
        fpft.spines_off(ax[-1])
    
    ax_γ = plt.subplot(gs[:,-1])
    box = ax_γ.boxplot(np.vstack([coef_γ_vs_freq[ds][:,-1] for ds in which_ds]).T, patch_artist=True,
                       widths=0.25, labels = [labels[dsi] for dsi in which_ds],
                       whis = [5,95],
                       
    )
    ax_γ.set_ylabel("$\Delta \gamma / \Delta f$", fontsize=11,labelpad=-1)
    [plt.setp(b, color = fpft.set_alpha(cols[ds],0.5), facecolor=fpft.set_alpha(cols[ds],0.5)) for ds, b in zip(which_ds, box["boxes"])]
    [plt.setp(b, color = cols[ds], linewidth=1) for ds, b in zip(which_ds, box["medians"])]
    fpft.spines_off(ax_γ)
    plt.tight_layout(w_pad=0)
    return ax, ax_γ
    

                                       
def plot_information_regression(data, which_ds, iprb,
                                which_log10_dists = defaultdict(lambda: [-1,0, np.log10(2)], {"16Ts":[np.log10(i) for i in [1e-1, 4e-1, 7e-1]]}),
                                mean_normalize = True,
                                same_plot      = True,
                                coef_plot_width = 2,
                                names  = {"s=p":"Surrogate data", "bw":"Simulations", "16Ts":"Supplementary"},
                                cols   = {"s=p":cm.gray(0.4), "bw":cm.GnBu(0.75), "16Ts":cm.GnBu(0.35), },                                
                                figsize = None,
                                plot_ils = False,
):
    gs = GridSpec(len(which_ds), len(which_log10_dists[which_ds[0]])+coef_plot_width)
    plt.figure(figsize=(8,2.5 * len(which_ds)) if figsize is None else figsize)
    ax = []
    for i, ds in enumerate(which_ds):
        ax.append([])
        F = data[ds]
        I = F.I[iprb][0]
        I_freqs = F.freqs[:I.shape[0]]
        ind_use = (I_freqs >= F.reg_freq_range[0]) & (I_freqs <= F.reg_freq_range[1])
        xx = I_freqs[ind_use].to(UNITS.Hz).magnitude
        xe = np.linspace(min(xx),max(xx),101)    
        d_scale = F.pitch.to(UNITS.um).magnitude        
        dd = F.I_dists/d_scale
        which_idists = []
        for j, ddj in enumerate(which_log10_dists[which_ds[0]]):
            which_idist = np.argmin(abs(dd - 10**ddj))
            which_idists.append(which_idist)
            ax[i].append(plt.subplot(gs[i,j]))
            axij = ax[i][-1]
            
            yy = I[ind_use, which_idist]
            c0,c1 = F.reg_coefs[iprb][0][which_idist]
            ye = 10**(c0 + c1 * xe)
            ym = 10**np.mean(np.log10(yy * d_scale**2))
            axij.semilogy(xx, (yy * d_scale**2)/(ym**mean_normalize),
                          "o", markersize=2,
                          label=f"FI / {ym:.2g}",
                          color = cols[ds],
            )
            axij.plot(xe, (ye * d_scale**2)/(ym**mean_normalize), "--",
                      color=cols[ds], linewidth=1,
                      label = f"$\\beta$" + f"={c1:1.1e}"
            )
            mean_normalize and axij.set_ylim(0.2,5)
            axij.legend(frameon=True, fontsize=8, labelspacing=0, handletextpad=0.5)
            xt = list(axij.get_xticks())
            if xx[0] not in xt: xt = ([xx[0]] + xt[1:]) if xt[0] < xx[0] else ([xx[0]] + xt)
            axij.set_xticks(xt)
            axij.set_xlim(0, xx[-1]+1)
            if mean_normalize:
                yt = np.array(list(np.arange(0.2,1,0.1)) + list(range(1,6)))
                axij.set_yticks(yt, labels= [s if s[-1] in "125" else "" for s in [f"{yti:g}" for yti in yt]], fontsize=10)
            [axij.spines[w].set_visible(False) for w in ["right", "top"]]
            not gs[i,j].is_first_col() and axij.set_yticklabels([])
            gs[i,j].is_last_row() and axij.set_xlabel("Frequency (Hz)")
            gs[i,j].is_first_col() and axij.set_ylabel("Fisher information")
            
        if same_plot and gs[i,j].is_first_row(): ax_coef = plt.subplot(gs[:,-coef_plot_width:])
        elif not same_plot: ax_coef = plt.subplot(gs[i,-coef_plot_width:])
        else: pass # ax_coef will already have been defined.
        pc = np.percentile(F.reg_coefs[iprb][1:][:,:,-1],[5,50,95],axis=0)
        ax_coef.semilogx(dd, pc[1], "o-", color=cols[ds], markersize=4, label = names[ds])
        ax_coef.fill_between(dd, pc[0], pc[2],color=fpft.set_alpha(cols[ds],0.1))
        if plot_ils and hasattr(F, "sim0") and hasattr(F.sim0, "integral_length_scales"):
            keys = list(F.sim0.integral_length_scales.keys())
            origin_key = (0 * UNITS.m, 0 * UNITS.m, "y")
            probe_key  = tuple([k for k in keys if k[-1] == "y" and k != origin_key][0])
            ils_origin = F.sim0.integral_length_scales[origin_key]["l"].to(F.pitch)
            ils_probe  = F.sim0.integral_length_scales[probe_key]["l"].to(F.pitch)
            #ax_coef.axvline(ils_origin.magnitude, ymin=0.48, ymax=0.52, color="r", linewidth=2, label="ILS (origin)")
            #ax_coef.axvline(ils_probe.magnitude,  ymin=0.48, ymax=0.52, color="r", linewidth=2, label="ILS (probe)")
            ax_coef.scatter([ils_origin.magnitude], [0], zorder = 5, c="orangered", marker="4", s = 150, linewidth=1.5,label="ILS (origin)")
            ax_coef.scatter([ils_probe.magnitude],  [0], zorder = 5, c="orangered",   marker="3", s = 150, linewidth=1.5,      label="ILS (probe)")
            print(f"{ils_origin=}")
            print(f"{ils_probe=}")            
            
        ax_coef.set_ylim(-0.05,0.05)
        ax_coef.set_ylabel("$\\beta = \Delta \log_{10}($FI$)/\Delta f$",labelpad=-8, fontsize=12)
        ax_coef.set_xlabel(f"Intersource distance ({pitch_sym})",labelpad=0, fontsize=12)
        [ax_coef.spines[w].set_visible(False) for w in ["right", "top"]]
    ax_coef.grid(True, linestyle=":")
    ax_coef.legend(frameon=False, labelspacing =0, loc="lower left")
    plt.tight_layout(pad=0,w_pad=0)
    
    for i, axi in enumerate(ax):
        for j, axij in enumerate(axi):
            trans = mtrans.Affine2D().translate(-0.06*j,0).scale(sx=1.25,sy=1)
            bbox  = axij.get_position()
            axij.set_position(mtrans.TransformedBbox(bbox, trans))

    return ax, ax_coef
    

    
    
