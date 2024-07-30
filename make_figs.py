#!/usr/bin/env python
import os, sys, yaml, logging
from importlib import reload
from argparse import ArgumentParser
from builtins import sum as bsum
import gc,resource

import utils
import units; reload(units); UNITS = units.UNITS;

logger = utils.create_logger(__name__)
INFO = logger.info
DEBUG = logger.debug
WARN = logger.warning

available_single = ["plumes_demo", "corr_decomp", "phase_example",
                    "mvg_fits", "mvg_supp_fits",
                    "scattergrams", "phase_heatmaps", "alap_fits", "rho_decay",
                    "fisher_info", "length_vs_freq", "elbow", "spectrum"]

available_plots = available_single + ["windowing", "ils", "multi_elbow","multi_decay_elbow", "multi_probes_geoms"]

parser = ArgumentParser()
parser.add_argument("which_figs", type=lambda x: x.split(","), default=[], help="Which figures to plot. Available: all, " + str(available_plots))
parser.add_argument('--datasets', type=lambda x: x.split(","), default=[], help="CSV file listing datasets to plot, or comma separated list of aliases.")
parser.add_argument('--surrogates', help="CSV file listing surrogate datasets.")
parser.add_argument('--window_length',  type=str, help="Window length to use.", default="1*UNITS.sec")
parser.add_argument('--window_shape',   type=str,  help="Window shape to use.", default="kaiser_9")
parser.add_argument("--fitk", action="store_true", help="Fit k.")
parser.add_argument("--dontfitb", action="store_true", help="Don't fit k.")
parser.add_argument("--figsize", type=str, default="(8,3)", help="Figure size.")
parser.add_argument("--iprb", type=int, default=0, help="Index of probe to use.")
parser.add_argument("--x_coords", type=lambda x: [float(xi) for xi in x.split(",")], default=[], help="Load only probes with these x coordinates in meters.")
parser.add_argument("--y_coords", type=lambda x: [float(xi) for xi in x.split(",")], default=[], help="Load only probes with these y coordinates in meters.")
parser.add_argument("--n_cols", type=int, default=3, help="Number of columns in ProbesGeoms figure.")
args = parser.parse_args()

if len(args.x_coords)>0:
    INFO(f"Only loading probes with x coordinates {args.x_coords}.")
else:
    INFO("Loading all available probes regardless of x-coordinates.")

if len(args.y_coords)>0:
    INFO(f"Only loading probes with y coordinates {args.y_coords}.")
else:
    INFO("Loading all available probes regardless of y-coordinates.")

plots_list = available_single if ((len(args.which_figs)>0) and args.which_figs[0] == "all") else args.which_figs

for p in plots_list:
    if p not in available_plots:
        WARN(f"Plot '{p}' not available. Available plots: {available_plots}.")
        plots_list.remove(p)

if ("multi_elbow" in plots_list) and len(args.datasets)==0:
    raise ValueError("Need to specify datasets for multi_elbow plot.")

        
INFO(f"Plots to make: {plots_list}")
if len(plots_list)==0:
    INFO("No plots to make. Exiting.")
    sys.exit(0)

iprb = args.iprb
INFO(f"Using probe {iprb}.")

sim_names = {"bw":"boulder16", "bw_X":"boulder16streamwise", "bw_45":"boulder16_45deg",
             "16Ts":"n16Tslow", "16Ts_X":"n16Tslow_X", "16Ts_45":"n16Tslow_45deg"}

probe0 = {"bw":(0.45, 0.5) * UNITS.m,
          "bw_45":(0.45, 0.5) * UNITS.m,
          "bw_X":(0.45, 0.5) * UNITS.m,
          "16Ts":(1.0, 0.50) * UNITS.m,
          "16Ts_X":(1.0, 0.50) * UNITS.m,
          "16Ts_45":(1.0, 0.50) * UNITS.m,
          }
probe_name_ = lambda ds, coords: "0" if str(coords) == str(probe0[ds]) else (f"{coords[0].magnitude:.2f}" + "_" + f"{coords[1].magnitude:.2f}")
    
to_use = {}

if args.datasets is not None:
    for ds in args.datasets:
        if ds in sim_names:
            to_use[ds] = {"sim_name": sim_names[ds]}
        else:
            assert os.path.exists(ds), f"Dataset file {ds} does not exist."
            with open(ds, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    line = [l.strip() for l in line.strip().split(",")]
                    if len(line)==4:
                        key, sim_name, probe_x, probe_y = line
                        to_use[key] = {"sim_name": sim_name, "which_coords": (float(probe_x) * UNITS.m, float(probe_y) * UNITS.m)}
INFO(f"Datasets = {to_use}.")

if args.surrogates is not None:                              
    assert os.path.exists(args.surrogates), f"Surrogate file {args.surrogates} does not exist."
    for line in open(args.surrogates, "r"):
        if line.startswith("#"):
            continue
        line = [l.strip() for l in line.strip().split(",")]
        if len(line)==4:
            key, sim_name, surrogate_k, random_seed = line
            to_use[key] = {"sim_name": sim_name, "surrogate_k": float(surrogate_k), "random_seed": int(random_seed)}
    
su_ds = [k for k,v in to_use.items() if v["sim_name"].startswith("surr")]
surrQ = lambda x: x in su_ds
surr_trialsQ = lambda x: any([x.startswith(s) for s in ["s=p_", "s=w_"]])
INFO(f"Surrogate datasets = {su_ds}.")

fit_k = args.fitk
fit_b = not args.dontfitb
window_shape  = args.window_shape
if "_" in window_shape:
    name, size = window_shape.split("_")
    window_shape = (name, int(size))
window_length = eval(args.window_length)

INFO(f"Window shape: {window_shape}")
INFO(f"Window length: {window_length}")
INFO(f"Fit k: {fit_k}")
INFO(f"Fit b: {fit_b}")

compute_filter = {
    "window_shape": window_shape,
    "window_length": window_length,
    "fit_k": fit_k,
    "fit_b": fit_b,
    "dmax_um": "1 * PITCH",
}
compute_surr = dict(**compute_filter); del compute_surr["dmax_um"]

INFO(f"Datasets: {to_use}")
INFO(f"Compute filter: {compute_filter}")

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

FisherPlumes = fp.FisherPlumes
crick.logger.setLevel(logging.DEBUG)
fp.logger.setLevel(logging.INFO)

is_surrogate = lambda x: (x["sim_name"].startswith("surr") or "surrogate_k" in x)

# Load datasets
if not all(["multi" in k for k in plots_list]):
    [f.logger.setLevel(logging.WARN) for f in [crick, boulder,fp]];
    proc.logger.setLevel(logging.INFO)
    loaded = {}
    for k, v in to_use.items():
        payload, matches = proc.load_data(strict = surrQ(k),
                                 init_filter = v,
                                 compute_filter = compute_filter if not surrQ(k) else compute_surr,
                                 # fit_corrs = ["search.1"],
                                 fit_corrs = [],
                                          return_matches = True,
                                          x_coords = args.x_coords,
                                          y_coords = args.y_coords,
                                 )
        assert payload is not None, f"No data loaded for {k}."
#        if len(payload) == 1:
#            loaded[k] = payload[0]
#            INFO(f"Loaded {k}.")
#        else:
        for p,m in zip(payload, matches):
            name = k
            
            if not is_surrogate(m["init"]) and ("which_coords" in m["init"]):
                coords = m["init"]["which_coords"][0]
                print(f"{k=}, {coords=}")
                name = f"{k}__{probe_name_(k, coords)}"
            loaded[name] = p
            INFO(f"Loaded {name}.")
    
    data =  {k:FisherPlumes(d) for k,d in loaded.items() if d is not None}
    
else:
    INFO(f"Plotting multi plots only, so not loading data.")
    data = {}

[f.logger.setLevel(logging.INFO) for f in [crick, boulder,fp, proc]];    
INFO(f"Loaded keys: {list(data.keys())}")

SAVEPLOTS = True # Whether to actually make the plots
fit_k         = args.fitk

fig_dir_full        = fpft.get_fig_dir(window_shape = window_shape, window_length = window_length, fit_k = fit_k, fit_b = fit_b, create = True); DEBUG(f"{fig_dir_full=}")
fig_dir_wnd_shp_len = fpft.get_fig_dir(window_shape = window_shape, window_length = window_length, fit_k = None,  create = True); DEBUG(f"{fig_dir_wnd_shp_len=}")
fig_dir_wnd_shp     = fpft.get_fig_dir(window_shape = window_shape, window_length = None,          fit_k = None,  create = True); DEBUG(f"{fig_dir_wnd_shp=}")
fig_dir_top         = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = None,  create = True); DEBUG(f"{fig_dir_top=}")
fig_dir_fitk        = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = fit_k, create = True); DEBUG(f"{fig_dir_fitk=}")
fig_dir_fitkb       = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = fit_k, fit_b = fit_b, create = True); DEBUG(f"{fig_dir_fitkb=}")

snapshots_dir = defaultdict(lambda: None,
                            {"bw": os.path.join(boulder.data_root, "original", "saved-snapshots"),
                             "bw_X": os.path.join(boulder.data_root, "streamwise", "saved-snapshots"),
                             "bw_45": os.path.join(boulder.data_root, "45deg", "saved-snapshots"),
                             "16Ts": None,
                             "16Ts_X": None,
                             "16Ts_45": None,
                             })

DEFAULT   = "default"
isdefault = lambda x: type(x) is str and x == DEFAULT
all_but_bw = ["bw_X", "bw_45", "16Ts", "16Ts_X", "16Ts_45"]

Info = namedtuple('Info','name,color')
infos = {"16Ts": Info(name="Supp. dataset",               color = "dodgerblue"),
         "16Ts_X": Info(name="Supp. dataset (X)",               color = "dodgerblue"),
         "16Ts_45": Info(name="Supp. dataset (45 deg)",               color = "dodgerblue"),
         "bw":   Info(name="Main dataset",                color = "orangered"),
         "bw_X":   Info(name="Main dataset (streamwise)",                color = "orange"),
         "bw_45":   Info(name="Main dataset (45 deg)",                color = "orange"),         
         "s=p_0":  Info(name="Surrogate (all =)",     color = "pink"),
         "s=w":  Info(name="Surrogate (all =, white)",    color = "green"),
         "shw":  Info(name="Surrogate (high>low, white)", color = "silver"),
         "shp":  Info(name="Surrogate (high>low)",  color = "violet"),
         "s=w_q0":  Info(name="Surrogate (quad, ϕ=0, white)", color="blue"),
         "s=w_q1":  Info(name="Surrogate (quad, ϕ=π/3, white)", color="green"),
}

which_srcs    = dict_update_from_field({"bw":[0,15], #[-3750, 3750],                                       
                                        "bw_X": [0, 15], # [-48750, 48750],
                                        "bw_45":[0, 15], #[-48749, 48749],
                                        "16Ts":[0,15], #[496000,504000],
                                        "16Ts_X":[0,15], #[16000,104000],
                                        "16Ts_45":[0,15], #[16000, 104000],
                                        },       
                                       su_ds, "bw")

class FigPlumesDemo:
    def __init__(self, data):
        self.data = data

        self.t_wnd         = dict_update_from_field({"bw":[-4,4]*UNITS.sec}, su_ds + all_but_bw, "bw")
        self.which_idists  = dict_update_from_field({"bw":[0,2,3]}, su_ds + all_but_bw, "bw")
        self.tticks        = dict_update_from_field({"bw":DEFAULT}, su_ds + all_but_bw, "bw")
        self.xticks        = dict_update_from_field({"bw":DEFAULT}, su_ds + all_but_bw, "bw")
        self.yticks        = dict_update_from_field({"bw":DEFAULT}, su_ds + all_but_bw, "bw")
        self.snapshot_time = defaultdict(lambda: 40000*UNITS.ms, {"16Ts":40010*UNITS.ms, "16Ts_X":40010*UNITS.ms, "16Ts_45":40010*UNITS.ms})
    
    def plot(self):
        INFO("\nPLOTTING FIGURES SHOWING EXAMPLE PLUME AND CORRELATIONS.")
        for kfull, F in sorted(self.data.items()):
            k = kfull.split("__")[0] # Probe locations are tacked on to the name as "__probe_locs"
            if surrQ(k): continue
            ax_plume, ax_traces, ax_corr = fpf.plot_plumes_demo(F,
                                                                self.snapshot_time[k],
                                                                which_srcs[k],
                                                                t_center = (self.snapshot_time[k].to(UNITS.ms).magnitude//1000)*1000 * UNITS.ms,
                                                                y_lim = (0,5.01) if not surrQ(k) else (-3.01,3.01),
                                                                y_ticks = [-3,0,3] if surrQ(k) else None,
                                                                data_dir = snapshots_dir[k],
                                                                mean_subtract_y_coords = "16" in k,
                                                                t_wnd = self.t_wnd[k],
                                                                dt = 1 * UNITS.sec,
                                                                which_idists=self.which_idists[k],
                                                                plot_source_locations = {"s":20, "edgecolor":"k", "c":"w","marker":"o", "which_sources":which_srcs[k]},
                                                            )
            
            not isdefault(self.tticks[k]) and ax_traces[-1].set_xticks(self.tticks[k])    
            not isdefault(self.xticks[k]) and ax_plume.set_xticks(self.xticks[k])
            not isdefault(self.yticks[k]) and ax_plume.set_yticks(self.yticks[k])
            if surrQ(k) or k  in ["bw"]: ax_corr.set_xticks(np.arange(5))
            if surrQ(k): [ax_corr.set_ylim(-0.85,1.05), ax_corr.set_ylabel("Correlation",labelpad=-8)]
            fpft.label_axes([ax_plume, ax_traces[0], ax_corr], "ABC", y = [0.99]*3, fontsize=12, fontweight="bold")
            file_name = f"{fig_dir_wnd_shp_len}/plumes_demo_{kfull}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
        
("plumes_demo" in plots_list) and FigPlumesDemo(data).plot()

class FigWindowing:
    def __init__(self):
        self.order = ["bw", "s=p_0", "shp", "bw"]
        self.which_wnd = [(1 * UNITS.s, 'hann')] * 3 + [(1 * UNITS.s, 'boxcar')]
        self.init_filter = {"bw": {"sim_name": "boulder16", "which_coords": (0.45 * UNITS.m, 0.5 * UNITS.m)},
                       "s=p_0":{"sim_name": "surr_all_equal", "surrogate_k":4, "random_seed":0},
                       "shp":{"sim_name": "surr_high",
                              "surrogate_k":4,
                              "random_seed":0
                              },
                       }
        compute_filter = []
        for i, (o,w) in enumerate(zip(self.order, self.which_wnd)):
            c = dict(**compute) if o == "bw" else dict(**compute_surr)
            c["window_length"], c["window_shape"] = w[0], w[1]
            compute_filter.append(c)

            
        
        self.data_wnd = [FisherPlumes(proc.load_data(strict = True,
                                                init_filter=self.init_filter[o],
                                                compute_filter = cf,
                                                     )[0],
                                      load_sims=False) for cf,o,wnd in zip(compute_filter, self.order,self.which_wnd)]        

    def plot(self):
        plt.figure(figsize=(8,5))
        gs = GridSpec(2,2)
        axes, cbs = [], []
        for (o, gsi, datai) in zip(self.order, gs, self.data_wnd):
            axes.append(plt.subplot(gsi))
            axes[-1], cbi = fpf.plot_fisher_information_heatmap(datai, 0, ax = axes[-1], freq_max = 25 * UNITS.Hz,
                                                                heatmap_range =[-2, np.log10(500)],
                                                                heatmap_cm    =cm.Spectral_r,
                                                                do_colorbar   = gsi.is_last_col(),
            )
            if gsi.is_first_row():
                axes[-1].tick_params(labelbottom=False)
                axes[-1].set_xlabel("")
        
            if not gsi.is_first_col():
                axes[-1].tick_params(labelleft = False)
                axes[-1].set_ylabel("")
        plt.tight_layout(w_pad = 1.5, h_pad=1.5)
        fpft.label_axes(axes, "ABCD",
                        align_y = [[0,1],[2,3]],
                        align_x = [[0,2],[1,3]],
                        fontsize=12, fontweight="bold", dy=0.01, dx = -0.01)
        file_name = f"{fig_dir_fitkb}/fisher_info_heatmaps.pdf"
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight', pad_inches=0), flush(f"Wrote {file_name}."));
        sys.stdout.flush(); plt.show(); plt.close();

("windowing" in plots_list) and FigWindowing().plot()        

class FigCorrDecomp():
    def __init__(self):
        self.xlims = defaultdict(lambda: DEFAULT)
        self.xticks = defaultdict(lambda: DEFAULT)
        self.which_freqs = defaultdict(lambda: [1,2,5,10] * UNITS.Hz)

    def plot(self):
        print("\nPLOTTING FIGURES SHOWING THE CORRELATION DECOMPOSITION.")
        for kfull, F in data.items():
            k = kfull.split("_")[0]
            if k.startswith("s=p"):
                if not k == "s=p_0":
                    continue
            which_freqs = self.which_freqs[k]
            labs = [f"{f}" for f in which_freqs]
            cols = {"All":cm.gray(0.4)}; cols.update({l:col for l,col in zip(labs, [cm.cool(1 - f.magnitude/10) for f in which_freqs])})    
            INFO(f"Plotting correlation decomposition for {k}.")
            slices = {"All":slice(1,10000)}
            freq_inds = F.freqs2inds(which_freqs)
            INFO(f"Mapped frequencies {which_freqs} to indices {freq_inds}.")
            slices.update({l:slice(fi, fi+1) for l, fi in zip(labs, freq_inds)})
            ax = fpf.plot_correlations(F.rho[iprb], F.pitch.to("um").magnitude, slices=slices, cols=cols, n_rows = 2, plot_order = ["All"] + labs)
            [(axi.set_xlabel(f"Intersource distance ({fpf.pitch_sym})"),
              not isdefault(self.xlims[k])  and axi.set_xlim(self.xlims[k]),
              not isdefault(self.xticks[k]) and axi.set_xticks(self.xticks[k])) for axi in ax]    
            file_name = f"{fig_dir_wnd_shp_len}/corr_components_{kfull}.pdf"
            fpft.label_axes(ax, "ABCDEF", fontsize=12, fontweight="bold", dy=-0.01)
            ax[-1].set_ylim(-0.5,1)
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."))
            sys.stdout.flush(); plt.show(); plt.close();
    
("corr_decomp" in plots_list) and FigCorrDecomp().plot()

class FigPhaseExample:
    def __init__(self):
        self.which_freq   = defaultdict(lambda: 5 * UNITS.Hz)
        self.which_idists = defaultdict(lambda: 1)
        self.fig_size = (8,3)

    def plot(self):
        print("\nPLOTTING PHASE RELATIONSHIPS EXAMPLE.")
        for fname, F in data.items():
            name = fname.split("__")[0]
            #if name != "bw" or not "16" in name: continue
            if surrQ(name): continue
            plt.figure(figsize=self.fig_size)
            which_freq = self.which_freq[name]
            ifreq = F.freqs2inds([which_freq])[0]
            idist = self.which_idists[name]
            axes = [plt.subplot(1,4,i+1) for i in range(4)]
            plt.sca(axes[0])
            fpf.plot_gm(sc=4.5, scale=[0.1,0.125],dxy=[0,0])    
            axes[0].axis("square")
            axes[0].set_ylim([0.41,0.59])
            ax_ = fpf.plot_a_vs_bcd(F, ifreq, idist, cols = [cm.cool(0.2), cm.cool(0.8), cm.cool(0.4)], al=[-0.5,0.5], ax = axes[1:])
            plt.tight_layout()
            fpft.label_axes(axes, "ABCD", fontsize=12, fontweight="bold", dy=-0.01, align_y=[[0,1,2,3]])            
            file_name = f"{fig_dir_wnd_shp_len}/a_vs_bcd_{fname}_{which_freq.magnitude}Hz_{idist=}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
    
("phase_example" in plots_list) and FigPhaseExample().plot()

class FigMvgFits:
    def __init__(self):
        self.freqs_to_plot = defaultdict(lambda: [5 * UNITS.Hz, 10 * UNITS.Hz])
        self.which_freqs = dict_update_from_field({"bw":self.freqs_to_plot["bw"]},   su_ds + all_but_bw, "bw"); 
        self.which_idists= dict_update_from_field({"bw":[0,1,2,3]},   su_ds + all_but_bw, "bw"); 
        self.dcol_scales = dict_update_from_field({"bw":120000},  su_ds + all_but_bw, "bw");
        self.configs = ["a_c", "a_d"]
        self.cols    = [cm.cool(r) for r in [0.9, 0.4]]

    def plot(self):
        print("\nPLOTTING FIGURES SHOWING THE MULTIVARIATE GAUSSIAN FITS.")
        for fname, F in sorted(data.items()):
            name = fname.split("__")[0]
            if surr_trialsQ(name): continue
            for which_freq in self.which_freqs[name]:
                ifreq = F.freqs2inds([which_freq])[0]
                INFO(f"Mapped {which_freq} to index {ifreq}.")
                
                plt.figure(figsize=(12, 3 * len(self.configs)))
                axes = []
                for i, config in enumerate(self.configs):
                    idists = self.which_idists[name]
                    ax = [plt.subplot(len(self.configs), len(idists), i*len(idists) + j+1) for j in range(len(idists))]
                    ax_ = fpf.plot_coef1_vs_coef2(F,
                                                  ifreq,
                                                  config=config,
                                                  iprb=iprb,
                                                  i_pos_dists_to_plot = self.which_idists[name],
                                                  col = self.cols[i],
                                                  axes = ax,
                                                  do_corr = True,
                    )
                    if i:
                        [axi.set_title("") for axi in ax_]
                                
                    axes.extend(ax)
                fpft.label_axes(axes, "ABCDEFGH", fontsize=12, fontweight="bold", dy=-0.01)            
                file_name = f"{fig_dir_wnd_shp_len}/coef_vs_coef_{fname}_{which_freq.magnitude}Hz_{'__'.join(self.configs)}.pdf"
                SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
                sys.stdout.flush(); plt.show(); plt.close();
    
("mvg_fits" in plots_list) and FigMvgFits().plot()

class FigMvgSuppFits:
    def __init__(self):
        self.freq      = dict_update_from_field({"bw":5 * UNITS.hertz,},    su_ds + all_but_bw, "bw") 
        self.idists    = dict_update_from_field({"bw":[0,1,2,3,4,6,7,12]},  su_ds + all_but_bw, "bw") 
        self.t_lim     = dict_update_from_field({"bw":[35, 45]*UNITS.sec},  su_ds + all_but_bw, "bw")
        self.dt        = dict_update_from_field({"bw":1*UNITS.sec},         su_ds + all_but_bw, "bw")

    def plot(self):
        print("\nPLOTTING SUPPLEMENTARY FIGURES SHOWING THE MULTIVARIATE GAUSSIAN FITS.")
        for kfull, F in sorted(data.items()):
            k = kfull.split("__")[0]
            if surrQ(k): continue
            plt.figure(figsize=(12,6))
            coef_ax, trace_ax = fpf.plot_coef_vs_coef_and_traces(F, self.freq[k], self.idists[k],
                                                                 which_probe = iprb, n_per_row = 2,
                                                                 y_lim=[0,5] if k[:2]!="su" else [-3,3],
                                                                 t_lim = self.t_lim[k],
                                                                 dt = self.dt[k])
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
            file_name = f"{fig_dir_wnd_shp_len}/coefs_and_traces_{kfull}_{self.freq[k].to(UNITS.hertz).magnitude}Hz.png" # Use png as these figures have lots of points
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
("mvg_supp_fits" in plots_list) and FigMvgSuppFits().plot()

class FigScattergrams:
    def __init__(self):
        self.freqs_to_plot = defaultdict(lambda: [2 * UNITS.Hz, 5 * UNITS.Hz, 8 * UNITS.Hz, 10 * UNITS.Hz])
        self.which_freqs   = dict_update_from_field({"bw":self.freqs_to_plot["bw"]},   su_ds + all_but_bw, "bw"); 
        self.dcol_scales = dict_update_from_field({"bw":120000},  su_ds + all_but_bw, "bw");

    def plot(self):
        print("\nPLOTTING SCATTERGRAMS.")
        for fname, F in sorted(data.items()):
            name = fname.split("__")[0]
            if surr_trialsQ(name): continue
            for which_freq in self.which_freqs[name]:
                ifreq = F.freqs2inds([which_freq])[0]
                INFO(f"Mapped {which_freq} to index {ifreq}.")
                ax = fpf.plot_scattergram(F,
                                          ifreq,
                                          iprb,
                                          figsize=(8,8),
                                          dist_col_scale = self.dcol_scales[name],
                                          markersize = 0.2,
                                          cols = ["royalblue","crimson", "seagreen", "magenta"],
                                          coef_names = {0:"Sin", 1:"Cos"},
                                          lim_scale = 2.,
                                          print_fun = np.corrcoef,
                                          )
                file_name = f"{fig_dir_wnd_shp_len}/scattergram_{fname}_{which_freq.magnitude}Hz.pdf"
                SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
                sys.stdout.flush(); plt.show(); plt.close();
    
("scattergrams" in plots_list) and FigScattergrams().plot()

class FigPhaseHeatmaps:
    def plot(self):
        print("\nPLOTTING PHASE HEATMAPS.")
        for fname, F in sorted(data.items()):
            name = fname.split("__")[0]
            if surr_trialsQ(name): continue
            ax = fpf.plot_phase_heatmap(F, max_phi = np.pi/3, plot_which="all", max_corr=0.1,figsize=(8,6))
            plt.tight_layout()
            fpft.label_axes(ax, "ABCD", 
                            align_y = [[0,1],[2,3]],
                            align_x = [[0,2],[1,3]],
                            fontsize=12, fontweight="bold", dy=-0.025)
            file_name = f"{fig_dir_wnd_shp_len}/phase_heatmap_{fname}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
    
("phase_heatmaps" in plots_list) and FigPhaseHeatmaps().plot()

class FigAlapFits:
    def __init__(self):
        self.idist     = dict_update_from_field({"bw":[0,1,2]},          su_ds + all_but_bw, "bw")
        self.freq_max  = dict_update_from_field({"bw":21 * UNITS.hertz}, su_ds + all_but_bw, "bw")
        self.vmin      = dict_update_from_field({"bw":[0,0]},            su_ds + all_but_bw, "bw")
        self.vmax      = dict_update_from_field({"bw":[1,1]},            su_ds + all_but_bw, "bw")
        self.fit_corrs = defaultdict(lambda: None)
    
    def plot(self):
        print("\nPLOTTING ASYMMETRIC LAPLACIAN FITS.")
        for fname, F in sorted(data.items()):
            name = fname.split("__")[0]
            if surrQ(name): continue
            if surr_trialsQ(name): continue
            d = np.array(list(F.rho[iprb].keys()))
            d = np.sort(d[d>=0])
            for f, xl in zip([1,5,10] * UNITS.hertz, [[-0.25, 1.0], [-0.02, 0.05], [-0.02, 0.05]]):
                if f != 5 * UNITS.hertz: continue
                which_freq = defaultdict(lambda: f)
                ax_cdf, ax_dcdf, ax_hm = fpf.plot_alaplace_fits(F, d[self.idist[name]],
                                                                which_probe = iprb,
                                                                ifreq_lim = [1, F.freqs2inds([self.freq_max[name]])[0]],
                                                                which_ifreq = F.freqs2inds([which_freq[name]])[0],
                                                                figsize=(9,4),
                                                                fit_color="gray",
                                                                vmax=self.vmax[name],
                                                                vmin=self.vmin[name],
                                                                plot_dvals=True,
                                                                expansion = 1.0,
                                                                xl = xl,
                                                                fit_corrs = self.fit_corrs[name],
                                                                leg_loc = None,
                                                                leg_loc2 = "lower right",
                                                                cdf_mode = "even")
                plt.tight_layout(pad=0)
                fpft.label_axes(ax_cdf + ax_dcdf + ax_hm, "ABCDEFGHIJK",
                                align_y = [[0,1,2,6],[3,4,5,7]],
                                align_x = [[0,3],[1,4],[2,5],[6,7]],
                                fontsize=12, fontweight="bold", dy=0)
                file_name = f"{fig_dir_wnd_shp_len}/alap_fits_{fname}_{which_freq[name].to(UNITS.hertz).magnitude}Hz.pdf"
                SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
                sys.stdout.flush(); plt.show(); plt.close();

("alap_fits" in plots_list) and FigAlapFits().plot()

class FigRhoDecayFits:
    def __init__(self):
        self.freqs  = dict_update({fld:[2,3,7,10] * UNITS.hertz for fld in ["bw", "16Ts", "16Ts_X", "16Ts_45", "bw_X","bw_45"]}, su_ds, [[1,3,17,20] * UNITS.hertz]*4)
        self.xl     = dict_update_from_field({"bw":(-10,200)},                 su_ds + all_but_bw, "bw"); 
        self.xt     = dict_update_from_field({"bw":np.arange(0,201,50)},          su_ds + all_but_bw, "bw"); 
        self.xtp    = dict_update_from_field({"bw":np.array([60,90,135])},     su_ds + all_but_bw, "bw"); 
        self.ytp    = dict_update_from_field({"bw":np.array([0.8,1,1.2,1.5])}, su_ds + all_but_bw, "bw");
    
    def plot(self):
        print("\nPLOTTING RHO DECAY FITS.")
        for kfull, F in sorted(data.items()):
            k = kfull.split("__")[0]
            if surrQ(k): continue
            ax = fpf.plot_la_gen_fits_vs_distance(F, 
                                                  figsize=(8,4), legloc = 'right',
                                                  log_scale = True,
                                                  scatter_size=1.5,
                                                  max_bs = 10,
                                                  which_ifreqs = F.freqs2inds(self.freqs[k]))
            [((i>1) and axi.set_xlabel(f"Intersource Distance $s$ ({fpf.pitch_sym})")) for i, axi in enumerate(ax[:4])]
            plt.tight_layout(h_pad=1,w_pad=0.5)
            fpft.label_axes(ax, "ABCDEFGHIJK",
                            align_y = [[0,1,4],[2,3]],
                            align_x = [[0,2],[1,3]],
                            fontsize=12, fontweight="bold", dy=-0.02)                        
            file_name = f"{fig_dir_full}/rho_vs_s_fits_{kfull}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
    
("rho_decay" in plots_list) and FigRhoDecayFits().plot()    

class FigFisherInfo:
    def __init__(self):
        self.freqs     = dict_update_from_field({"bw":[1,2, 5, 10, 20] * UNITS.hertz}, su_ds + all_but_bw, "bw")
        self.freq_max  = dict_update_from_field({"bw":25 * UNITS.hertz},               su_ds + all_but_bw, "bw")
        self.colscale  = dict_update_from_field({"bw":10},                             su_ds + all_but_bw, "bw")
        self.d_vals_um = dict_update_from_field({"bw":[1,5,50]},                       su_ds + all_but_bw, "bw")
        self.d_lim_um  = dict_update_from_field({"bw":[100, 125000 ]},                 su_ds + all_but_bw, "bw")
        self.bf_ytick  = dict_update_from_field({"bw":[0,5,10]},                       su_ds + all_but_bw, "bw")
        self.bf_yl     = dict_update_from_field({"bw":[0,15]},                         su_ds + all_but_bw, "bw")
        self.plot_param_fits = False

    def plot(self):
        print("\nPLOTTING FISHER INFORMATION.")
        for kfull, F in sorted(data.items()):
            k = kfull.split("__")[0]
            prefix = k.split(".")[0]
            if prefix not in ["bw","16Ts", "16Ts_X", "16Ts_45", "bw_X","bw_45"]: continue
            plt.figure(figsize=(6,7))
            ax_fisher, ax_best_freq, ax_d = fpf.plot_fisher_information(F,
                                                                        which_probe = iprb,
                                                                        d_lim_um   = self.d_lim_um[k],
                                                                        d_vals_um  = np.array(self.d_vals_um[k])*1000,
                                                                        d_space_fun  = lambda d0,d1,n:np.logspace(np.log10(d0),np.log10(d1),n),
                                                                        which_ifreqs = F.freqs2inds(self.freqs[k]),
                                                                        x_stagger = lambda x, i: x*(1.02**i),
                                                                        plot_fun = plt.loglog,
                                                                        log_scale = True,
                                                                        plot_param_fits = self.plot_param_fits,
                                                                        freq_max  = self.freq_max[k],
                                                                        colfun    = lambda f: cm.cool_r(f/self.colscale[k]),
                                                                        info_heatmap = True,
                                                                        heatmap_range =[-2, np.log10(500)],
                                                                        heatmap_cm    =cm.Spectral_r,
            )
            ax_fisher.set_ylim(1e-2,1e3)
            plt.tight_layout(h_pad=2,w_pad=0)
            fpft.label_axes([ax_fisher, ax_best_freq] + ax_d , "ABCDEFGHIJK",
                            #align_y = [[2,3,4]],
                            align_x = [[0,1,2] if self.plot_param_fits else [0,1]],
                            fontsize=12, fontweight="bold", dy=-0.02)
    
            file_name = f"{fig_dir_full}/fisher_info_{kfull}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
("fisher_info" in plots_list) and FigFisherInfo().plot()

def get_paired_ds(paired_ds, data):
    if paired_ds is None:
        paired_ds = []
    else:
        if paired_ds not in data:
            WARN(f"Paired dataset {paired_ds} not loaded so not including.")
            paired_ds = []
        else:
            paired_ds = [paired_ds]
    return paired_ds

class FigLengthVsFreq:
    def __init__(self):
        self.which_corr_freqs_Hz = defaultdict(lambda: [2, 5, 10, 20])
        self.paired_ds = defaultdict(lambda: "s=p_0")

    def plot(self):
        print("\nPLOTTING LENGTH CONSTANTS VS FREQUENCY.")                
        for kfull, F in sorted(data.items()):
            k = kfull.split("__")[0]
            prefix = k.split(".")[0]
            if prefix not in ["16Ts", "16Ts_X", "16Ts_45", "bw_X","bw_45", "bw"]:
                continue            
            paired_ds = get_paired_ds(self.paired_ds[k], data)
            which_ds = [kfull] + paired_ds
            ax, ax_γ = fpf.plot_length_constants_vs_frequency(data, which_ds, iprb, which_corr_freqs_Hz = self.which_corr_freqs_Hz[k])
            fpft.label_axes(ax + [ax_γ], "ABCDE",
                            fontsize=12, fontweight="bold",
                            dx = -0.01, dy=0.01,
                            align_x = [[0,2],[1,3]],
                            align_y = [[0,1,4]])
            
            file_name = f"{fig_dir_full}/length_vs_freq_{kfull}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
("length_vs_freq" in plots_list) and FigLengthVsFreq().plot()    


class FigElbow:
    def __init__(self):
        self.other_ds = defaultdict(lambda: [f"s=p_{i}" for i in range(10)])

    def plot(self):
        print("\nPLOTTING ELBOW.")
        for kfull, F in sorted(data.items()):
            k = kfull.split("__")[0]
            if surrQ(k): continue
            which_ds = bsum([get_paired_ds(ods, data) for ods in self.other_ds[k]],[])
            if len(which_ds) == 0:
                WARN(f"No paired datasets for {k}. Skipping plot.")
                continue

            names = {ki:ki for ki in which_ds}
            names[k] = k
            # cols = {k:cm.hsv(i/(len(which_ds))) for i,k in enumerate(which_ds)}
            datai = {}
            datai[k] = F
            for ki in which_ds:
                datai[ki] = data[ki]
            ax, ax_coef = fpf.plot_information_regression(datai,
                                                          [k] + which_ds,
                                                          iprb,
                                                          plot_ils = True,
                                                          do_label = [True]*2 + [False]*(len(which_ds)-1),
                                                          yl=(-0.06,0.06))
        
            
            fpft.label_axes([ax[0][0], ax[1][0], ax_coef], "ABC",
                            fontsize=12, fontweight="bold",
                            align_x = [[0,1]],
                            align_y = [[0,2]])
        
            file_name = f"{fig_dir_full}/reg_coefs_{kfull}.pdf"
            #tight_layout(pad=0,w_pad=0, h_pad=0)
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
("elbow" in plots_list) and FigElbow().plot()

class FigMultiElbow:
    t_snap = lambda ds: (40 + (0.01)*("16" in ds)) * UNITS.s

    @staticmethod
    def get_xylims(ds):
        ylims= [-0.06, 0.06]
        # if ds == "bw_X":
        #     ylims = [0, 0.06]
        #     if window_length == 0.5 * UNITS.sec:
        #         ylims = [0.01, 0.06]
    
        xlims = [1e-3, 5] if "16" in ds else [4e-3, 1e1]
        return xlims, ylims

    @staticmethod
    def plot_elbows(D, ax_elbow, ax_plume = None):
        xlims, ylims = FigMultiElbow.get_xylims(ds)
    
        for j,(k,F) in enumerate(sorted(D.items())):
            x,y = F.used_probe_coords[0]
            x_p = x.to(F.pitch).magnitude
            y_p = y.to(F.pitch).magnitude
            dy = ((F.y_lim[1] + F.y_lim[0])/2).to(F.pitch).magnitude
            col = cm.tab10(j) if j < 10 else cm.Set3(j-10)
            (ax_plume is not None) and ax_plume.plot(x_p, y_p - dy, "x", markersize=10, color=col, markeredgewidth=2)
            fpf.plot_elbow(F, ax = ax_elbow, error_bars = False, markerstyle = "-", col=col, lw=2)
            ax_elbow.axvline(x=1, color="gray", linestyle="dotted", lw=0.5, zorder=-1)
            ax_elbow.axhline(y=0, color="gray", linestyle="dotted", lw=0.5, zorder=-1)

            ax_elbow.set_ylim(ylims[0]-0.01, ylims[1]+0.01)
        ax_elbow.set_yticks([ylims[0],0,ylims[1]])
        ax_elbow.set_xlim(xlims)
        
        ytlabs = ax_elbow.get_yticklabels()
        ytlabs[1] = "0"
        ax_elbow.set_yticklabels(ytlabs)
        # Set the top and right spines invisible
        [ax_elbow.spines[spine].set_visible(False) for spine in ["top", "right"]]
        ax_elbow.set_ylabel("$\\beta$", fontsize=12, labelpad=-20)
        ax_elbow.set_xlabel(f"Intersource distance ({fpf.pitch_sym})",    labelpad=0,   fontsize=10)

        return ax_elbow    
       
    @classmethod
    def plot(cls, which_ds, ax_elbow = None):
        n_ds = len(which_ds)
        
        plt.figure(figsize=(8, 2.2 * n_ds))
        loaded = {}
        gs = GridSpec(n_ds, 2, width_ratios=[1, 1])
        ax = []
        for i, ds in enumerate(which_ds):            
            # Load the data for this compute_filter, and for all the coords in the probe_locs
            init_filter = {"sim_name":sim_names[ds]}
            matches = proc.find_registry_matches(init_filter = {"sim_name":sim_names[ds]},
                                                 compute_filter = compute_filter,
                                                 x_coords = args.x_coords,
                                                 y_coords = args.y_coords,
                                                 )

            assert len(matches) > 0, f"Found no matches for {ds}."
            print(f"Found {len(matches)} matches for {ds}.")

            ds_base = ds.split("_")[0]
            loaded[ds] = {}
            srcs = [np.mod(w,16) for w in which_srcs[ds]]
            for m in matches:
                print(m)
                if "which_coords" not in m["init"]:
                    continue
                coords = m["init"]["which_coords"][0]
                probe_name = probe_name_(ds_base, coords)
                print(f"Loading data for {ds} at {probe_name}.")
                loaded[ds][probe_name] = utils.safe_load(proc.load_data(strict = True,
                                                                        init_filter = {"sim_name":init_filter["sim_name"],"which_coords":coords},
                                                                        compute_filter = compute_filter,
                                                                        load_sims = srcs if probe_name == "0" else [0],
                                                                        load_only = (["sims"] if probe_name == "0" else [])+ ['sim0', 'reg_coefs', 'I_dists', 'pitch_string', 'pitch'],
                                                                    ))
            assert "0" in loaded[ds], f"Could not find data for {ds} at probe location 0, found only {list(loaded[ds].keys())}."
                
            D = {}
            for k, d in loaded[ds].items():
                D[k] = FisherPlumes(d)
                D[k].used_probe_coords = D[k].sim0.get_used_probe_coords()
                D[k].y_lim = D[k].sim0.y_lim
                if k != "0":
                    del D[k].sim0
            
            ax_plume = plt.subplot(gs[i,0])
            ax.append(ax_plume)

            fpf.plot_plumes_snapshot(D["0"], FigMultiElbow.t_snap(ds), srcs, data_dir = snapshots_dir[ds], ax_plume = ax_plume);
    
            (i < n_ds - 1) and ax_plume.set_xlabel(None)
            ax_elbow = plt.subplot(gs[i,1])
            ax.append(ax_elbow)
            
            FigMultiElbow.plot_elbows(D, ax_elbow, ax_plume)
            (i != n_ds - 1) and ax_elbow.set_xlabel(None)

        plt.tight_layout()

        fpft.label_axes(ax, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", align_x = [list(range(0,len(ax),2)), list(range(1,len(ax),2))], align_y=list([i,i+1] for i in range(0,len(ax),2)), fontsize=12, fontweight="bold", dy=0.01)
        
        name = "_".join([d.replace("_","") for d in which_ds])
        fig_name = f"all_elbows_{name}.pdf"
        fig_full_path = os.path.join(fig_dir_full, fig_name)
        print(f"Saving figure to {fig_full_path}")
        plt.savefig(fig_full_path, bbox_inches="tight")
            
        return ax
("multi_elbow" in plots_list) and FigMultiElbow.plot(args.datasets)

class FigMultiDecayElbow:
    def __init__(self):
        self.t_snap = lambda ds: (40 + (0.01)*("16" in ds)) * UNITS.s
        self.which_corr_freqs_Hz = [2, 5, 10, 15, 20]
        self.which_corr_freqs = self.which_corr_freqs_Hz * UNITS.Hz
        self.labs = [f"{f}" for f in self.which_corr_freqs]
        self.cols = defaultdict(lambda: cm.gray(0.4), {"s=p":cm.gray(0.4), "bw":cm.GnBu(0.75), "bw.1_3":cm.GnBu(0.75), "bw_45":cm.GnBu(0.75),"bw_X":cm.GnBu(0.75), "16Ts":cm.GnBu(0.35), "16Ts_X":cm.GnBu(0.25), "16Ts_45":cm.GnBu(0.2)})
        self.cols.update({l:col for l,col in zip(self.labs, [cm.cool(1 - f.magnitude/20) for f in self.which_corr_freqs])})

    @staticmethod
    def map_coords_to_grid(ds):
        init_filter = {"sim_name":sim_names[ds]}
        matches = proc.find_registry_matches(init_filter = {"sim_name":sim_names[ds]},
                                             compute_filter = compute_filter,
                                             x_coords = args.x_coords,
                                             y_coords = args.y_coords,
                                             )
        assert len(matches) > 0, f"No matches found for {ds}."
        assert "init" in matches[0], f"No init data found for {ds}."
        assert "which_coords" in matches[0]["init"], f"No which_coords found for {ds}."
        
        coords = [m["init"]["which_coords"][0] for m in matches]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        x_set = sorted(list(set(x_coords)))
        y_set = sorted(list(set(y_coords)))
        n_rows = len(y_set)
        n_cols = len(x_set)
        gs_index = {}
        for c in coords:
            row = len(y_set) - 1 - y_set.index(c[1])
            col = x_set.index(c[0])
            key = str(c)
            gs_index[key] = (row, col)
            INFO(f"Mapping {c} to {gs_index[key]}")
        return n_rows, n_cols, gs_index
        
    def plot(self, which_ds, D = None):
        n_ds = len(which_ds)

        n_rows, n_cols, gs_index = {}, {}, {}
        for ds in which_ds:
            n_rows[ds], n_cols[ds], gs_index[ds] = self.map_coords_to_grid(ds)

        total_rows = sum(n_rows.values())
        col_width  = max(n_cols.values())
        total_cols = col_width * 3
            
        plt.figure(figsize=(8, 2.5 * n_ds))
        gs = GridSpec(total_rows, total_cols)
        loaded = {}
        ax = []
        irow, icol = 0, 0
        ax_corr = []
        for i, ds in enumerate(which_ds):            
            # Load the data for this compute_filter, and for all the coords in the probe_locs
            init_filter = {"sim_name":sim_names[ds]}
            matches = proc.find_registry_matches(init_filter = {"sim_name":sim_names[ds]},
                                                 compute_filter = compute_filter,
                                                 x_coords = args.x_coords,
                                                 y_coords = args.y_coords,
                                                 )

            assert len(matches) > 0, f"Found no matches for {ds}."
            print(f"Found {len(matches)} matches for {ds}.")

            
            ds_base = ds.split("_")[0]
            loaded = {}
            srcs = [0, -1] #[np.mod(w,16) for w in which_srcs[ds]]
            coords_for_probe = {}
            for m in matches:
                if "which_coords" not in m["init"]:
                    continue
                coords = m["init"]["which_coords"][0]                
                probe_name = probe_name_(ds_base, coords)
                coords_for_probe[probe_name] = coords
                print(f"Loading data for {ds} at {probe_name}.")
                loaded[probe_name] = utils.safe_load(proc.load_data(strict = True,
                                                                        init_filter = {"sim_name":init_filter["sim_name"],"which_coords":coords},
                                                                        compute_filter = compute_filter,
                                                                        load_sims = srcs if probe_name == "0" else [0],
                                                                        load_only = (["sims"] if probe_name == "0" else [])+ ['sim0', 'rho', 'coef_γ_vs_freq', 'pitch_string', 'pitch', 'fs', 'wnd', 'reg_coefs', 'I_dists', 'svals_um','source_line'],
                                                                    ))
                print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")
            assert "0" in loaded, f"Could not find data for {ds} at probe location 0, found only {list(loaded.keys())}."
                
            D = {}
            for k, d in loaded.items():
                print(f"Initializing FisherPlumes for {k}.")
                D[k] = FisherPlumes(d)
                D[k].used_probe_coords = D[k].sim0.get_used_probe_coords()
                D[k].y_lim = D[k].sim0.y_lim
                if k != "0":
                    del D[k].sim0
                print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")

            ax_plume = plt.subplot(gs[irow:irow+n_rows[ds],:col_width])
            ax.append(ax_plume)

            fpf.plot_plumes_snapshot(D["0"], self.t_snap(ds), srcs, data_dir = snapshots_dir[ds], ax_plume = ax_plume,
                                     plot_source_locations = {"which_sources":srcs,
                                                              "s":20, "c":"w", "marker":"o", "edgecolor":"k", "linewidth":1},
                                     );
    
            (i < n_ds - 1) and ax_plume.set_xlabel(None)
                    
            for j,(k,F) in enumerate(sorted(D.items())):
                x,y = F.used_probe_coords[0]
                x_p = x.to(F.pitch).magnitude
                y_p = y.to(F.pitch).magnitude
                dy = ((F.y_lim[1] + F.y_lim[0])/2).to(F.pitch).magnitude
                col = cm.tab10(j) if j < 10 else cm.Set3(j-10)
                ax_plume.plot(x_p, y_p - dy, "x", markersize=10, color=col, markeredgewidth=2)

                key    = str(coords_for_probe[k])
                ii, jj = gs_index[ds][key]
                # Set the padding the subplots to 0
                new_ax = plt.subplot(gs[ii + irow, jj + col_width])
                ax_corr.append((new_ax,ii,jj)) # Collect these for adjustment later
                
                freq_min = F.fs/F.wnd #(1/window_length.to(UNITS.s).magnitude) * UNITS.Hz        
                coef_γ_vs_freq = F.coef_γ_vs_freq[iprb]
                d_scale   = F.pitch.to(UNITS.um).magnitude
                freq_inds = F.freqs2inds(self.which_corr_freqs)
                slices    = {}    
                slices.update({l:slice(fi, fi+1) for l, fi in zip(self.labs, freq_inds)})    
                fpf.plot_correlations(F.rho[0], F.pitch.to(UNITS.um).magnitude, slices = slices, cols = self.cols,
                          plot_slices = False, plot_overlay=True, ax = [new_ax],
                                      plot_legend = False,
                                      tight_layout = False,
                                      )
                first_row = ii == 0
                last_row = ii == n_rows[ds] - 1
                first_col = jj == 0
                new_ax.set_xlabel("")

                # Set the xtics fontsize
                new_ax.tick_params(axis='x', labelsize=6)
                new_ax.tick_params(axis='y', labelsize=6)

                new_ax.set_title("")
                new_ax.set_yticks(np.arange(0,1.1,0.5))
                (not (first_col and last_row)) and new_ax.set_yticklabels([])
                (not (first_col and last_row)) and new_ax.set_xticklabels([])
                if first_row and first_col:
                    ax.append(new_ax)
                    
                if last_row and first_col:
                    new_ax.set_ylabel("Correlations", fontsize=8)
                    new_ax.set_xlabel(f"Intersource distance ({fpf.pitch_sym})", fontsize=8)
                    # Make the legend as tight as possible
                    new_ax.legend(loc="upper right", fontsize=5, ncol=1, frameon=False, labelspacing=0, handlelength=0.5)
                else:
                    new_ax.set_ylabel("")

                # Set the axis colors to col
                #[sp.set_color(col) for sp in new_ax.spines.values()]
                # Set the axis background color to col, but with alpha=0.5
                new_ax.set_facecolor(list(col)[:3] + [0.2])
                new_ax.axis("auto")

            left, bottom, width, height = 2*col_width/total_cols, (irow+n_rows[ds])/total_rows, col_width/total_cols, n_rows[ds]/total_rows
            ax_elbow = plt.subplot(gs[irow:irow+n_rows[ds],2*col_width:])
            #ax_elbow = plt.axes([left, bottom, width, height])
            FigMultiElbow.plot_elbows(D, ax_elbow)
            #ax_elbow.yaxis.tick_right()
            #ax_elbow.yaxis.set_label_position("right")
            ax.append(ax_elbow)
            irow += n_rows[ds]
            print(f"Memory usage after plotting {ds}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")
            gc.collect()
            print(f"After forcing garbage collection: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")
    
        plt.tight_layout(w_pad=0, h_pad=0, pad = 0)
        # Increase the width of the axes in ax_corr using set_position
        for i, (a,ii,jj) in enumerate(ax_corr):
            x0, y0, dx, dy = a.get_position().bounds
            a.set_position([x0 - 0.02*jj, y0-0.02*(n_rows[ds]-1-ii), dx*1.8, dy*1.5])

        fpft.label_axes(ax, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", align_x = [list(range(i,len(ax),3)) for i in range(3)], align_y=list([i,i+1,i+2] for i in range(0,len(ax),3)), fontsize=12, fontweight="bold", dy=0, dx = bsum([[0, -0.025,0]]*len(which_ds), []))

        name = "_".join([d.replace("_","") for d in which_ds])
        fig_name = f"all_corr_decays_{name}.pdf"
        fig_full_path = os.path.join(fig_dir_full, fig_name)
        print(f"Saving figure to {fig_full_path}")
        plt.savefig(fig_full_path, bbox_inches="tight")
            
        return ax
("multi_decay_elbow" in plots_list) and FigMultiDecayElbow().plot(args.datasets)

class FigMultiProbesGeoms:
    def __init__(self):
        self.t_snap = lambda ds: (40 + (0.01)*("16" in ds)) * UNITS.s
        
    def plot(self, which_ds, n_cols = 3):
        n_ds   = len(which_ds)
        n_rows = int(np.ceil(n_ds / n_cols))

        plt.figure(figsize=(8, 2.5 * n_rows))
        loaded = {}
        ax = []
        x_coords = lambda ds: [0.35, 0.4, 0.45] if "bw" in ds else [0.9, 1.0, 1.1]
        y_coords = lambda ds: [0.3, 0.5, 0.7]
        for i, ds in enumerate(which_ds):            
            # Load the data for this compute_filter, and for all the coords in the probe_locs
            init_filter = {"sim_name":sim_names[ds]}
            matches = proc.find_registry_matches(init_filter = {"sim_name":sim_names[ds]},
                                                 compute_filter = compute_filter,
                                                 x_coords = x_coords(ds),
                                                 y_coords = y_coords(ds),
                                                 )

            assert len(matches) > 0, f"Found no matches for {ds}."
            print(f"Found {len(matches)} matches for {ds}.")
            
            ds_base = ds.split("_")[0]
            loaded = {}
            #srcs = [np.mod(w,16) for w in which_srcs[ds]]
            srcs = [0, -1]
            coords_for_probe = {}
            for m in matches:
                if "which_coords" not in m["init"]:
                    continue
                coords = m["init"]["which_coords"][0]                
                probe_name = probe_name_(ds_base, coords)
                coords_for_probe[probe_name] = coords
                # Only load the data for the probe "0"
            
                print(f"Loading data for {ds} at {probe_name}.")
                loaded[probe_name] = utils.safe_load(proc.load_data(strict = True,
                                                                    init_filter = {"sim_name":init_filter["sim_name"],"which_coords":coords},
                                                                    compute_filter = compute_filter,
                                                                    load_sims = srcs if probe_name == "0" else [0],
                                                                    load_only = (["sims"] if probe_name == "0" else []) + ['sim0', 'pitch_string', 'pitch', 'svals_um', 'source_line'],
                                                                ))
                print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")
            assert "0" in loaded, f"Could not find data for {ds} at probe location 0, found only {list(loaded.keys())}."
                
            D = {}
            for k, d in loaded.items():
                print(f"Initializing FisherPlumes for {k}.")
                D[k] = FisherPlumes(d)
                D[k].used_probe_coords = D[k].sim0.get_used_probe_coords()
                D[k].y_lim = D[k].sim0.y_lim
                if k != "0":
                    del D[k].sim0
                print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")

            ax_plume = plt.subplot(n_rows, n_cols, i+1)
            ax.append(ax_plume)
            INFO(f"Plotting plumes from sources {srcs} for {ds}.")
            fpf.plot_plumes_snapshot(D["0"], self.t_snap(ds), srcs, data_dir = snapshots_dir[ds], ax_plume = ax_plume,
                                     plot_source_locations = {"which_sources":srcs,
                                                              "s":20, "c":"w", "marker":"o", "edgecolor":"k", "linewidth":1},
                                     );
    
            (i < n_rows - 1) and ax_plume.set_xlabel(None)
                    
            #for j,(k,F) in enumerate(sorted(D.items())):
            for j,(k,F) in enumerate(sorted(D.items())):
                x,y = F.used_probe_coords[0]                                
                x_p = x.to(F.pitch).magnitude
                y_p = y.to(F.pitch).magnitude
                dy  = ((F.y_lim[1] + F.y_lim[0])/2).to(F.pitch).magnitude
                col = cm.tab10(j) if j < 10 else cm.Set3(j-10)
                ax_plume.plot(x_p, y_p - dy, "x", markersize=10, color=col, markeredgewidth=2)

            print(f"Memory usage after plotting {ds}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")
    
        plt.tight_layout(w_pad=0, h_pad=0, pad = 0)

        fpft.label_axes(ax, "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        fontsize=12,
                        fontweight="bold",
#                        dy=0,
#                        dx = bsum([[0, -0.025,0]]*len(which_ds), []),
#                        align_x = [list(range(i,len(ax),n_cols)) for i in range(n_rows)],
#                        align_y = list(list([i + j] for j in range(n_cols)) for i in range(0,len(ax),3)),                        
                        )

        name = "_".join([d.replace("_","") for d in which_ds])
        fig_name = f"probe_geom_{name}.pdf"
        fig_dir_top = fpft.get_fig_dir()
        fig_full_path = os.path.join(fig_dir_top, fig_name)
        print(f"Saving figure to {fig_full_path}")
        plt.savefig(fig_full_path, bbox_inches="tight")
            
        return ax
("multi_probes_geoms" in plots_list) and FigMultiProbesGeoms().plot(args.datasets, args.n_cols)



class FigIls:
    def plot(self):
        print("\nPLOTTING ILS.")    
        for kfull, F in sorted(data.items()):
            k = kfull.split("_")[0]
            if surrQ(k): continue
            if not hasattr(F, "sim0"):
                WARN(f"Skipping {kfull} because it doesn't have a sim0 attribute.")
                continue
            if not hasattr(F.sim0, "integral_length_scales"):
                WARN(f"Skipping {kfull} because it doesn't have an integral_length_scales attribute.")
                continue
    
            ils = F.sim0.integral_length_scales
    
            keys = ils.keys()
            plt.figure(figsize=(8.5,3.5))
            ax = []
            for i, d in enumerate("xy"):
                ax.append(plt.subplot(1,2,i+1))
                orig_key = (0 * UNITS.m, 0 * UNITS.m, d)
                probe_key = [k for k in keys if k[-1]==d and k!=orig_key][0]
                for j, (kk, name) in enumerate(zip([orig_key,probe_key], ["origin", "probe"])):
                    res   = ils[kk]
                    fr, l = res["fr"], res["l"]
                    # l = nansum(fr) * ds
                    ds = l / np.nansum(fr)
                    xx = np.arange(len(fr))*ds.to(F.pitch).magnitude
                    plt.plot(xx, fr, label=f"p={name}")
                    lmag = l.to(F.pitch).magnitude
                    plt.gca().axvline(lmag, color=f"C{j}", linewidth=1, linestyle=":", label = f"$L_U$ = {lmag:.2} {fpf.pitch_sym}")
                plt.legend()
                plt.ylabel(f"$\langle \\widetilde u_{d}(p) \\widetilde u_{d}(p + r \hat e_{d}) \\rangle$")
                plt.xlabel(f"r ({fpf.pitch_sym})")
                plt.title(f"{d}-velocity autocorrelation function")
                fpft.spines_off(plt.gca())
            plt.tight_layout()
            fpft.label_axes(ax, "AB",
                                fontsize=12, fontweight="bold",
                                align_y = [[0,1]])
            file_name = f"figs/ils_supp_{kfull}.pdf"
            SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
            sys.stdout.flush(); plt.show(); plt.close();
("ils" in plots_list) and FigIls().plot()

class FigSpectrum:
    def plot(self):
        print("\nPLOTTING SPECTRA.")
        plt.figure(figsize=(8,5))
        for ki, (kfull, F) in enumerate(sorted(data.items(), key=lambda x: infos[x].name if x in infos else x)):
            kall = kfull.split("__")
            k = kall[0]
            k1 = f"({kall[1]})" if len(kall) > 1 else ""
            if k not in infos: continue
            f = []
            for _, s in F.stft.items():
                fr, tt, S = s[0]
                f.append(np.abs(S))
                
            fs = F.fs.to("Hz").magnitude
            f = np.array(f)
            a = np.mean(f,axis=-1).mean(axis=0)    
            plt.loglog(fr[fr<fs/2][1:],a[fr<fs/2][1:]/a[1] * (10**0),
                   label=infos[k].name + k1,
                   color=infos[k].color)
        plt.legend(borderpad=0)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized amplitude")
        plt.title("Plume spectra averaged over windows and source locations")
        plt.grid(True, which='both', linestyle=":")
        fig_dir = fig_dir_wnd_shp_len
        name = "_".join(args.datasets)
        file_name = f"{fig_dir}/spectra_{name}.pdf"
        SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
        sys.stdout.flush(); plt.show(); plt.close();
("spectrum" in plots_list) and FigSpectrum().plot()

print("ALLDONE")
