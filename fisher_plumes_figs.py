import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
plt.style.use("default")
from matplotlib import colors as mcolors
from matplotlib.gridspec   import GridSpec
from matplotlib import cm
from scipy.stats import mannwhitneyu,ttest_1samp

import fisher_plumes_fig_tools as fpft
import fisher_plumes_tools as fpt
from boulder import concs2rgb


scaled2col = lambda s, scale=1., cmap=cm.cool_r: cmap(s/scale)
dist2col   = lambda d, d_scale = 120000, cmap = cm.cool_r: scaled2col(d, d_scale, cmap)
freq2col   = lambda f, f_scale = 10,     cmap = cm.cool_r: scaled2col(f, f_scale, cmap)

def plot_plumes_demo(F, t_snapshot, 
                     which_keys,
                     data_dir = "./data",
                     which_idists = [0,1,2],
                     t_wnd = [-4,4],
                     y_lim = (0,3),
                     dt = 0.5
):
    fields = F.load_saved_snapshots(t = t_snapshot, data_dir = data_dir)
    plt.figure(figsize=(8,3))
    gs = GridSpec(3,3)
    ax_plume = plt.subplot(gs[:,0])
    pp = concs2rgb(fields[which_keys[0]], fields[which_keys[1]])
    ax_plume.matshow(pp, extent = F.sim0.x_lim + F.sim0.y_lim)
    px, py = F.sim0.get_used_probe_coords()[0]
    ax_plume.plot(px, py, "kx", markersize=5)
    ax_plume.xaxis.set_ticks_position('bottom')
    ax_plume.axis("auto")
    plt.xlabel("x (m)", labelpad=-1)
    plt.ylabel("y (m)", labelpad=-1)
    #ax_plume.set_yticks(arange(-0.2,0.21,0.1) if 'wide' in name else arange(-0.1,0.11,0.1))
    
    dists  = np.array(sorted(F.sims.keys()))
    middle = np.mean(dists)
    dists  = dists[dists>middle]
    t_lim  = np.array(t_wnd) + t_snapshot
    ax_trace = []
    for i, di in enumerate(which_idists):
        ax_trace.append(plt.subplot(gs[i,1]))
        d_mid = int(dists[di] - middle)
        a = F.sims[middle + d_mid].data.flatten()
        b = F.sims[middle - d_mid].data.flatten()
        t = F.sim0.t
        sc = max(a.std(), b.std())
        a /= sc
        b /= sc
        ax_trace[-1].plot(t,a,color="r", label=f"{middle + d_mid}", linewidth=1)
        ax_trace[-1].plot(t,b,color="b", label=f"{middle - d_mid}", linewidth=1)
        (i < 2) and ax_trace[-1].set_xticklabels([])
        (i ==2) and ax_trace[-1].set_xlabel("Time (sec.)", labelpad=-1)
        fpft.spines_off(ax_trace[-1])
        ax_trace[-1].set_xlim(*t_lim)
        ax_trace[-1].set_xticks(np.arange(t_lim[0],t_lim[-1]+0.01,dt))
        ax_trace[-1].set_ylim(*y_lim)
        ax_trace[-1].set_yticks(np.arange(min(y_lim),max(y_lim)+1,5))
        ax_trace[-1].tick_params(axis='both', labelsize=8)
        ax_trace[-1].set_ylabel("Conc.", labelpad=-1)
        #ax_trace[-1].legend(frameon=False,labelspacing=0,fontsize=6)
        wndf = lambda x: x[(t>=t_lim[0])*(t<t_lim[-1])]
        aw, bw = wndf(a), wndf(b)
        ρ_w = np.corrcoef(aw,bw)[0,1]
        ρ   = np.corrcoef(a, b)[0,1]
        # text(tlim[0], yl[1], f"$\Delta$ = {2*dists[di]/1000:g} mm\n$\\rho$ = {ρ_w:1.3f} (window)\n$\\rho$ = {ρ:1.3f} (all)", fontsize=6, verticalalignment="top")
        plt.title(f"$\Delta$ = {2*d_mid/1000:g} mm, $\\rho_w$ = {ρ_w:1.3f}, $\\rho$ = {ρ:1.3f}", fontsize=8, verticalalignment="top")

        
    ax_corr_dist = plt.subplot(gs[:,-1])
    rho   = F.rho
    dists = np.array(sorted(list(rho.keys()))) 
    rho   = {d:rho[d][0] for d in dists} # Take the raw data, not the bootstraps
    rhom  = np.array([np.mean(np.sum(rho[d],axis=0)) for d in dists])
    rhos  = np.array([ np.std(np.sum(rho[d],axis=0)) for d in dists])
    col   = "gray"
    plt.fill_between(dists/1000, rhom-rhos,rhom+rhos, color=fpft.set_alpha(mpl.colors.to_rgba(col),0.2));
    fpft.pplot(dists/1000, rhom , "o-", markersize=4,color=col);
    ax_corr_dist.grid(True, linestyle=":")
    #ax_corr_dist.set_yticklabels(["-1","","0","","1"])
    ax_corr_dist.set_ylabel("Correlation",labelpad=-1)
    ax_corr_dist.set_xlabel("Intersource distance (mm)", labelpad=-1)
    plt.tight_layout(pad=0,w_pad=0,h_pad=1)

    return ax_plume, ax_trace, ax_corr_dist


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
                        dist_col_scale = 120000,
                        
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

    dist2col_ = lambda d: dist2col(d,dist_col_scale)
    for i, d in enumerate(which_d):
        plt.subplot(1,3,i+1)
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
        plt.title(f"{d/1000:g} mm")
        plt.grid(True)    


def plot_alaplace_fits(F, which_dists,
                       ifreq_lim = [], xl = [-0.2,0.5], which_ifreq = 1, 
                       figsize=None, vmax=None,
                       cm_heat = cm.rainbow,
                       heatmap_xmax = np.inf,
                       heatmap_default_xticks = False,
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
    (not heatmap_default_xticks) and plt.gca().set_xticks(pdists/1000)
    plt.xticks(rotation=45, fontsize=8)    
    plt.xlabel("Distance (mm)")
    plt.ylabel("Frequency (Hz)", labelpad=-1)
    plt.title("Mismatch",pad=-2)
    plt.axis("auto")
    plt.colorbar()
    pdmax = pdists[np.argmin(np.abs(pdists - 100*1000))]
    plt.xlim((pdists[0]-dd/2)/1000,min((pdmax+dd/2)/1000, heatmap_xmax)) #pdists[-1]/1000-0.5)

    plt.tight_layout(h_pad = 0, w_pad=0.2) #, w_pad = 0)
        
def plot_gen_exp_parameter_fits_panel(F, which_fis, contours_dist = None,
                                      d_scale = 1,
                                      n_contours = 12, contours_cmap=cm.gray,
                                      plot_scatter = True,
                                      plot_legend = True,
                                      scatter_alpha= 0.8, scatter_size=3,
                                      colfun = lambda f: freq2col(f, 10, cmap=cm.cool_r)
):
    γbs = F.fit_params[:, which_fis, 1]/d_scale
    kbs = F.fit_params[:, which_fis, 2]
    freqs = np.arange(F.wnd)/F.wnd*F.fs

    if plot_scatter:
        for γ, k in zip(γbs, kbs):
            plt.scatter(γ, k,
                        c=[fpft.set_alpha(colfun(f), scatter_alpha) for i, f in enumerate(freqs[which_fis])],
                        s = scatter_size, zorder=10)    
    
    hmus = []
    for ifreq, fi in enumerate(which_fis):
        γk = np.array([np.array(γbs[:,ifreq]), np.array(kbs[:,ifreq])])
        mu = np.mean(γk,axis=1)
        C  = np.cov(γk)
        hmui, hsdsi = fpft.plot_bivariate_gaussian(mu, C,
                                                   n_sds = 1, n_points = 1000,
                                                   mean_style = {"marker":"o", "markersize":3, "color":colfun(freqs[fi])},
                                                   sd_style   = {"linewidth":1, "color":colfun(freqs[fi])})
        hmus.append(hmui)

        plot_legend and plt.legend(handles = hmus, labels=[f"{freqs[fi]:g} Hz" for fi in which_fis],
               labelspacing=0,
               frameon=False,
               fontsize=6,
               loc='lower left'
    )
    
    xl = plt.xlim()
    yl = plt.ylim()

    if n_contours:
        contours_dist_mm = np.array(contours_dist)/d_scale
        γγ, kk = np.meshgrid(np.linspace(*xl, 101), np.linspace(*yl,101))
        I = 2*np.log10(kk) - kk*np.log10(γγ) + (kk - 2) * np.log10(contours_dist_mm)
        plt.contourf(γγ, kk, I, n_contours, cmap=contours_cmap)  

    fpft.spines_off(plt.gca())
    plt.ylabel("Exponent $k_n$")
    plt.xlabel("Length Scale $\gamma_n$ (mm)")
    plt.title("Fit Parameters")

        
    
def plot_la_gen_fits_vs_distance(F, 
                                 d_scale = 1000, which_ifreqs = [1,2,3,4],
                                 figsize = None, legloc = None, xl = None,
                                 colfun = lambda f: freq2col(f, 10)
):

    (figsize is not None) and plt.figure(figsize=figsize)

    gs     = GridSpec(2,3)
    dd_all = np.array(sorted(list(F.la.keys())))
    la     = {d:F.la[d][0] for d in F.la}
    la_sub = np.array(la[d] for d in dd_all if d in F.dd_fit)
    freqs  = np.arange(F.wnd)/F.wnd*F.fs
    ax     = []
    max_norm = lambda y: y/np.max(y)
    identity = lambda y: y

    dx = dd_all/d_scale    
    for i, fi in enumerate(which_ifreqs):
        row   = i // 2
        col   = i % 2
        ax.append(plt.subplot(gs[row, col]))
        la_mean = np.array([np.mean(F.la[d][1:, fi]) for d in dd_all])
        la_sd   = np.array([np.std(F.la[d][1:, fi]) for d in dd_all])
        ax[-1].plot(dx, la_mean, "o:", color=colfun(fi), linewidth=1, markersize=2, label=f"{freqs[fi]:g} Hz")
        ax[-1].plot([dx,dx], [la_mean - la_sd, la_mean+la_sd], "-", color=fpft.set_alpha(colfun(fi),0.5), linewidth=1)
        for j in range(min(5, F.n_bootstraps)):
            ax[-1].plot(F.dd_fit/d_scale, fpt.gen_exp(F.dd_fit, *(F.fit_params[1+j][fi])), color=fpft.set_alpha(colfun(fi),0.5), linewidth=1)
        (row == 1) and plt.xlabel("Distance $s$ (mm)")
        (col == 0) and plt.ylabel("$\lambda_n(s)$")        
        (xl is not None) and plt.xlim(xl)
        plt.title(f"{freqs[fi]:g} Hz", pad=-2)
        fpft.spines_off(plt.gca())

    # PLOT THE PARAMETERS
    ax.append(plt.subplot(gs[:,-1]))
    plot_gen_exp_parameter_fits_panel(F, which_ifreqs, d_scale = d_scale, n_contours = 0)
    
    fpft.spines_off(plt.gca())
    plt.ylabel("Exponent $k_n$")
    plt.xlabel("Length Scale $\gamma_n$ (mm)")
    plt.title("Fit Parameters")
    return ax
    
def plot_fisher_information(#amps, sds, slope, intercept,
        F,
        d_lim=[1e-3,100],
        d_vals = [0.1,1,10],
        d_range = None,
        d_scale = 1,
        d_space_fun = np.linspace,
        which_ifreqs = [2,4,6,8,10],
        x_stagger = lambda x, i: x*(1.02**i),
        ifreq_to_freq = 1,
        fi_scale = 1000,
        plot_fun = plt.plot,
        colfun = lambda f: cm.cool_r(f/10.)        
):

    d_ranges = [d_lim] if d_range is None else [d_range]
    assert len(d_ranges)==1, f"Expected only one d_range, got {len(d_ranges)=}."

    n_dvals = len(d_vals)
    gs      = GridSpec(5,n_dvals)

    for i, d in enumerate(d_vals):
        ax = plt.subplot(gs[3:,i])
        plot_gen_exp_parameter_fits_panel(F, np.arange(1,11), contours_dist = d,
                                          d_scale = d_scale,
                                          n_contours = 12, contours_cmap=cm.gray,
                                          plot_legend = (i==0),
                                          plot_scatter = False,
                                          colfun = colfun)
        ax.set_title(f"{d/d_scale:g} mm")
        (i != 0) and (ax.set_ylabel(""), ax.set_yticklabels([]))
            
    for i, (d0, d1) in enumerate(d_ranges):
        d = d_space_fun(d0, d1,11)

        Ibs = F.compute_fisher_information_at_distances(d).transpose([2,1,0]) # freq x bs x dist
        Ibs  *= d_scale**2 # To get it in units of mm^{-2}
        Imean = np.mean(Ibs, axis=1) # Bootstrap mean
        Istd  = np.std (Ibs, axis=1)

        plt.subplot(gs[:3,:])        
        for i, (fi, Im, Is) in enumerate(zip(which_ifreqs, Imean[which_ifreqs], Istd[which_ifreqs])):
            x = x_stagger(d/d_scale,i)
            plot_fun(x, Im, "o-", linewidth=1,markersize=2,color=colfun(F.freqs[fi]), label = f"{fi * ifreq_to_freq:g} Hz")
            plot_fun([x, x], [Im-Is*3, Im+Is*3], color = fpft.set_alpha(colfun(F.freqs[fi]),0.5), linewidth=1)

        Isort = np.argsort(Imean[1:],axis=0) # [1:] to skip DC
        best_freqs        = Isort[-1] + 1
        second_best_freqs = Isort[-2] + 1
        p_vals = np.array([ttest_1samp(Ibs[best_freq, :, di] -  Ibs[second_best_freq,:,di],0, alternative='greater')[1]
                  for di, (best_freq, second_best_freq) in enumerate(zip(best_freqs, second_best_freqs))])
        for i, (bf,p,Im) in enumerate(zip(best_freqs, p_vals, Imean[best_freqs[0]])):
            if bf == best_freqs[0]:
               n_stars = int(np.floor(-np.log10(p)))
               di = x_stagger(d[i]/d_scale,bf)
               plt.text(di, Im, "*"*min(n_stars,3), fontsize=12)
    
    plt.legend(frameon=False, labelspacing=0.25,fontsize=8)
    plt.ylabel("Fisher Information (mm$^{-2}$)" + (f"x {fi_scale}" if fi_scale != 1 else ""))
    plt.xlabel("Distance (mm)", labelpad=-1)
    fpft.spines_off(plt.gca())
        

    plt.tight_layout(pad=0)

    
