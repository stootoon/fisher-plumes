import os, sys, yaml, logging
from importlib import reload
from argparse import ArgumentParser


import utils
import units; reload(units); UNITS = units.UNITS;

logger = utils.create_logger(__name__)
INFO = logger.info
DEBUG = logger.debug


parser = ArgumentParser()
parser.add_argument('datasets', help="CSV file listing datasets to plot.")
parser.add_argument('--window_length',  type=str, help="Window length to use.", default="1*UNITS.sec")
parser.add_argument('--window_shape',   type=str,  help="Window shape to use.", default="('kaiser',9)")
parser.add_argument("--fitk", action="store_true", help="Fit k.")
parser.add_argument("--dontfitb", action="store_true", help="Don't fit k.")
parser.add_argument("--figsize", type=str, default="(8,3)", help="Figure size.")
parser.add_argument("--iprb", type=int, default=0, help="Index of probe to use.")
parser.add_argument("--plot_only", type=lambda x: x.split(","), default=[], help="Plot only these figures.")
args = parser.parse_args()

if len(args.plot_only):
    plots_list = args.plot_only
else:
    plots_list = ["plumes_demo"]

INFO(f"Plots to make: {plots_list}")

iprb = args.iprb
INFO(f"Using probe {iprb}.")

assert os.path.exists(args.datasets), f"Dataset file {args.dataset} does not exist."
to_use = {}
with open(args.datasets, "r") as f:
    for line in f:
        line = [l.strip() for l in line.strip().split(",")]
        if len(line)==4:
            key, sim_name, probe_x, probe_y = line
            to_use[key] = {"sim_name": sim_name, "which_coords": (float(probe_x) * UNITS.m, float(probe_y) * UNITS.m)}


compute = {"window_shape": eval(args.window_shape),
           "window_length": eval(args.window_length),
           "fit_k": args.fitk,
           "fit_b": not args.dontfitb,
           "dmax_um": "1 * PITCH",
           }

INFO(f"Datasets: {to_use}")
INFO(f"Compute: {compute}")

import numpy as np
import re,pickle
from builtins import sum as bsum
from collections import defaultdict, namedtuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.transforms as mtransforms
plt.style.use("default")
plt.rcParams["figure.figsize"] = eval(args.figsize)
flush = lambda *args, **kwargs: [sys.stdout.flush(), print(*args, **kwargs)]

import boulder
import crick
import surrogate
import fisher_plumes_tools as fpt
import fisher_plumes_figs  as fpf
import fisher_plumes_fig_tools as fpft
import fisher_plumes as fp
from utils import dict_update, dict_update_from_field
import proc

import fig_params
fig_params.logger.setLevel(logging.DEBUG)

FisherPlumes = fp.FisherPlumes
crick.logger.setLevel(logging.DEBUG)
fp.logger.setLevel(logging.INFO)

# Load datasets
[f.logger.setLevel(logging.WARN) for f in [crick, boulder,fp]];
loaded = {k:utils.safe_load(proc.load_data(strict = True,
                                     init_filter = v,
                                     compute_filter = compute,
                                     # fit_corrs = ["search.1"],
                                     fit_corrs = [],                                     
                                     ))          
          for k,v in to_use.items()}

data =  {k:FisherPlumes(d) for k,d in loaded.items() if d is not None}

[f.logger.setLevel(logging.INFO) for f in [crick, boulder,fp]];
su_ds = [k for k,v in to_use.items() if v["sim_name"].startswith("surr")]
surrQ = lambda x: x in su_ds
surr_trialsQ = lambda x: any([x.startswith(s) for s in ["s=p_", "s=w_"]])
INFO(f"Surrogate datasets = {su_ds}.")

SAVEPLOTS = True # Whether to actually make the plots
FigParams = fig_params.FigParams(UNITS, compute, su_ds)

isdefault = fig_params.isdefault

def fig__plumes_demo():
    INFO("\nPLOTTING FIGURES SHOWING EXAMPLE PLUME AND CORRELATIONS.")
    P = FigParams.plumes_demo
    for k, F in sorted(data.items()):
        if surrQ(k): continue
        ax_plume, ax_traces, ax_corr = fpf.plot_plumes_demo(F,
                                                            P.snapshot_time[k],
                                                            P.which_srcs[k],
                                                            t_center = (P.snapshot_time[k].to(UNITS.ms).magnitude//1000)*1000 * UNITS.ms,
                                                            y_lim = (0,5.01) if not surrQ(k) else (-3.01,3.01),
                                                            y_ticks = [-3,0,3] if surrQ(k) else None,
                                                            data_dir = P.snapshots_dir[k],
                                                            mean_subtract_y_coords = "16" in k,
                                                            t_wnd = P.t_wnd[k],
                                                            dt = 1 * UNITS.sec,
                                                            which_idists=P.which_idists[k],
                                                            plot_source_locations = {"s":20, "edgecolor":"k", "c":"w","marker":"o", "which_sources":P.which_srcs[k]},
                                                        )
        
        not isdefault(P.tticks[k]) and ax_traces[-1].set_xticks(P.tticks[k])    
        not isdefault(P.xticks[k]) and ax_plume.set_xticks(P.xticks[k])
        not isdefault(P.yticks[k]) and ax_plume.set_yticks(P.yticks[k])
        if surrQ(k) or k  in ["bw"]: ax_corr.set_xticks(np.arange(5))
        if surrQ(k): [ax_corr.set_ylim(-0.85,1.05), ax_corr.set_ylabel("Correlation",labelpad=-8)]
        fpft.label_axes([ax_plume, ax_traces[0], ax_corr], "ABC", y = [0.99]*3, fontsize=12, fontweight="bold")
        file_name = f"{FigParams.fig_dir_wnd_shp_len}/plumes_demo_{k}.pdf"
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
        sys.stdout.flush(); plt.show()
    
("plumes_demo" in plots_list) and fig__plumes_demo()


def fig__corr_decomp():
    print("\nPLOTTING FIGURES SHOWING THE CORRELATION DECOMPOSITION.")
    P = FigParams.corr_decomp
    for k, F in data.items():
        if k.startswith("s=p"):
            if not k == "s=p_0":
                continue
        which_freqs = P.which_freqs[k]
        labs = [f"{f}" for f in which_freqs]
        cols = {"All":cm.gray(0.4)}; cols.update({l:col for l,col in zip(labs, [cm.cool(1 - f.magnitude/10) for f in which_freqs])})    
        INFO(f"Plotting correlation decomposition for {k}.")
        slices = {"All":slice(1,10000)}
        freq_inds = F.freqs2inds(which_freqs)
        INFO(f"Mapped frequencies {which_freqs} to indices {freq_inds}.")
        slices.update({l:slice(fi, fi+1) for l, fi in zip(labs, freq_inds)})
        ax = fpf.plot_correlations(F.rho[iprb], F.pitch.to("um").magnitude, slices=slices, cols=cols, n_rows = 2, plot_order = ["All"] + labs)
        [(axi.set_xlabel(f"Intersource distance ({fpf.pitch_sym})"),
          not isdefault(P.xlims[k])  and axi.set_xlim(P.xlims[k]),
          not isdefault(P.xticks[k]) and axi.set_xticks(P.xticks[k])) for axi in ax]    
        file_name = f"{FigParams.fig_dir_wnd_shp_len}/corr_components_{k}.pdf"
        fpft.label_axes(ax, "ABCDEF", fontsize=12, fontweight="bold", dy=-0.01)
        ax[-1].set_ylim(-0.5,1)
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."))
        sys.stdout.flush(); plt.show()

("corr_decomp" in plots_list) and fig__corr_decomp()

def fig__phase_example():
    print("\nPLOTTING PHASE RELATIONSHIPS EXAMPLE.")
    P = FigParams.phase_example
    for name, F in data.items():
        #if name != "bw" or not "16" in name: continue
        if surrQ(name): continue
        plt.figure(figsize=P.fig_size)
        which_freq = P.which_freq[name]
        ifreq = F.freqs2inds([which_freq])[0]
        idist = P.which_idists[name]
        axes = [plt.subplot(1,4,i+1) for i in range(4)]
        plt.sca(axes[0])
        fpf.plot_gm(sc=4.5, scale=[0.1,0.125],dxy=[0,0])    
        axes[0].axis("square")
        axes[0].set_ylim([0.41,0.59])
        ax_ = fpf.plot_a_vs_bcd(F, ifreq, idist, cols = [cm.cool(0.2), cm.cool(0.8), cm.cool(0.4)], al=[-0.5,0.5], ax = axes[1:])
        plt.tight_layout()
        fpft.label_axes(axes, "ABCD", fontsize=12, fontweight="bold", dy=-0.01, align_y=[0,1,2,3])            
        file_name = f"{FigParams.fig_dir_wnd_shp_len}/a_vs_bcd_{name}_{which_freq.magnitude}Hz_{idist=}.pdf"
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
        sys.stdout.flush(); plt.show()

("phase_example" in plots_list) and fig__phase_example()        

def fig__mvg_fits():
    print("\nPLOTTING FIGURES SHOWING THE MULTIVARIATE GAUSSIAN FITS.")
    P = FigParams.mvg_fits
    for name, F in sorted(data.items()):
        if surr_trialsQ(name): continue
        for which_freq in P.which_freqs[name]:
            ifreq = F.freqs2inds([which_freq])[0]
            INFO(f"Mapped {which_freq} to index {ifreq}.")
            
            plt.figure(figsize=(12, 3 * len(P.configs)))
            axes = []
            for i, config in enumerate(P.configs):
                idists = P.which_idists[name]
                ax = [plt.subplot(len(P.configs), len(idists), i*len(idists) + j+1) for j in range(len(idists))]
                ax_ = fpf.plot_coef1_vs_coef2(F,
                                              ifreq,
                                              config=config,
                                              iprb=iprb,
                                              i_pos_dists_to_plot = P.which_idists[name],
                                              col = P.cols[i],
                                              axes = ax,
                                              do_corr = True,
                )
                if i:
                    [axi.set_title("") for axi in ax_]
                            
                axes.extend(ax)
            fpft.label_axes(axes, "ABCDEFGH", fontsize=12, fontweight="bold", dy=-0.01)            
            file_name = f"{FigParams.fig_dir_wnd_shp_len}/coef_vs_coef_{name}_{which_freq.magnitude}Hz_{'__'.join(P.configs)}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show()

("mvg_fits" in plots_list) and fig__mvg_fits()

def fit__scattergrams():
    print("\nPLOTTING SCATTERGRAMS.")
    P = FigParams.scattergrams
    for name, F in sorted(data.items()):
        if surr_trialsQ(name): continue
        for which_freq in P.which_freqs[name]:
            ifreq = F.freqs2inds([which_freq])[0]
            INFO(f"Mapped {which_freq} to index {ifreq}.")
            ax = fpf.plot_scattergram(F,
                                      ifreq,
                                      iprb,
                                      figsize=(8,8),
                                      dist_col_scale = P.dcol_scales[name],
                                      markersize = 0.2,
                                      cols = ["royalblue","crimson", "seagreen", "magenta"],
                                      coef_names = {0:"Sin", 1:"Cos"},
                                      lim_scale = 2.,
                                      print_fun = np.corrcoef,
                                      )
            file_name = f"{FigParams.fig_dir_wnd_shp_len}/scattergram_{name}_{which_freq.magnitude}Hz.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show()

("scattergrams" in plots_list) and fit__scattergrams()


def fit__phase_heatmaps():
    print("\nPLOTTING PHASE HEATMAPS.")
    for name, F in sorted(data.items()):
        if surr_trialsQ(name): continue
        ax = fpf.plot_phase_heatmap(F, max_phi = np.pi/3, plot_which="all", max_corr=0.1,figsize=(8,6))
        plt.tight_layout()
        fpft.label_axes(ax, "ABCD", 
                        align_y = [[0,1],[2,3]],
                        align_x = [[0,2],[1,3]],
                        fontsize=12, fontweight="bold", dy=-0.025)
        file_name = f"{FigParams.fig_dir_wnd_shp_len}/phase_heatmap_{name}.pdf"
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));

("phase_heatmaps" in plots_list) and fit__phase_heatmaps()

def fig__alap_fits():
    print("\nPLOTTING ASYMMETRIC LAPLACIAN FITS.")
    P = FigParams.alap_fits
    for name, F in sorted(data.items()):
        if surrQ(name): continue
        if surr_trialsQ(name): continue
        d = np.array(list(F.rho[iprb].keys()))
        d = np.sort(d[d>=0])
        for f, xl in zip([1,5,10] * UNITS.hertz, [[-0.25, 1.0], [-0.02, 0.05], [-0.02, 0.05]]):
            if f != 5 * UNITS.hertz: continue
            which_freq = defaultdict(lambda: f)
            ax_cdf, ax_dcdf, ax_hm = fpf.plot_alaplace_fits(F, d[P.idist[name]],
                                                            which_probe = iprb,
                                                            ifreq_lim = [1, F.freqs2inds([P.freq_max[name]])[0]],
                                                            which_ifreq = F.freqs2inds([which_freq[name]])[0],
                                                            figsize=(9,4),
                                                            fit_color="gray",
                                                            vmax=P.vmax[name],
                                                            vmin=P.vmin[name],
                                                            plot_dvals=True,
                                                            expansion = 1.0,
                                                            xl = xl,
                                                            fit_corrs = P.fit_corrs[name],
                                                            leg_loc = None,
                                                            leg_loc2 = "lower right",
                                                            cdf_mode = "even")
            plt.tight_layout(pad=0)
            fpft.label_axes(ax_cdf + ax_dcdf + ax_hm, "ABCDEFGHIJK",
                            align_y = [[0,1,2,6],[3,4,5,7]],
                            align_x = [[0,3],[1,4],[2,5],[6,7]],
                            fontsize=12, fontweight="bold", dy=0)
            file_name = f"{FigParams.fig_dir_wnd_shp_len}/alap_fits_{name}_{which_freq[name].to(UNITS.hertz).magnitude}Hz.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show()
("alap_fits" in plots_list) and fig__alap_fits()


def fig__mvg_supp_fits():
    print("\nPLOTTING SUPPLEMENTARY FIGURES SHOWING THE MULTIVARIATE GAUSSIAN FITS.")
    P = FigParams.mvg_supp_fits
    for k, F in sorted(data.items()):
        if surrQ(k): continue
        plt.figure(figsize=(12,6))
        coef_ax, trace_ax = fpf.plot_coef_vs_coef_and_traces(F, P.freq[k], P.idists[k],
                                                             which_probe = iprb, n_per_row = 2,
                                                             y_lim=[0,5] if k[:2]!="su" else [-3,3],
                                                             t_lim = P.t_lim[k],
                                                             dt = P.dt[k])
        for ax in coef_ax:
            ax.set_xlabel("")
            ax.set_ylabel("")
        [ax.legend(fontsize=6,labelspacing=0,frameon=False) for ax in trace_ax]
        plt.tight_layout(pad=0)
        all_ax = bsum([[ax_c, ax_t] for ax_c, ax_t in zip(coef_ax, trace_ax)], [])
        n_ax   = len(all_ax)
        fpft.label_axes(all_ax,
                        [ch+nu for ch in "ABCDEFGH" for nu in "12"],
                        align_x = [list(range(i,n_ax,4)) for i in range(4)],
                        align_y = [list(range(i,i+4)) for i in range(0,n_ax,4)],
                        fontsize=12, fontweight="bold", dy=-0.01)
        file_name = f"{FigParams.fig_dir_wnd_shp_len}/coefs_and_traces_{k}_{P.freq[k].to(UNITS.hertz).magnitude}Hz.png" # Use png as these figures have lots of points
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
        sys.stdout.flush(); plt.show()
("mvg_supp_fits" in plots_list) and fig__mvg_supp_fits()
exit(0)
