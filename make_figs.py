import os, sys, yaml, logging
from importlib import reload
from argparse import ArgumentParser


import utils
import units; reload(units); UNITS = units.UNITS;

logger = utils.create_logger(__name__)
INFO = logger.info
DEBUG = logger.debug


parser = ArgumentParser()
parser.add_argument('datasets', help="YAML file listing datasets to plot.")
parser.add_argument('--window_length',  type=str, help="Window length to use.", default="1*UNITS.sec")
parser.add_argument('--window_shape',   type=str,  help="Window shape to use.", default="('kaiser',9)")
parser.add_argument("--dontfitk", action="store_true", help="Don't fit k.")
parser.add_argument("--figsize", type=str, default="(8,3)", help="Figure size.")
args = parser.parse_args()

assert os.path.exists(args.datasets), f"Dataset file {args.dataset} does not exist."

compute = {"window_shape": eval(args.window_shape),
           "window_length": eval(args.window_length),
           "fit_k": not args.dontfitk}

window_shape  = compute["window_shape"]
window_length = compute["window_length"]
fit_k         = compute["fit_k"]

# Load datasets
with open(args.datasets, 'r') as f:
    to_use = yaml.load(f, Loader=yaml.FullLoader)

INFO(f"Datasets: {to_use}")
INFO(f"Compute: {compute}")

import numpy as np
import re,pickle
from builtins import sum as bsum
from collections import defaultdict, namedtuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
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

# Load datasets
[f.logger.setLevel(logging.WARN) for f in [crick, boulder,fp]];

data =  {k:FisherPlumes(proc.load_data(init_filter = v,
                                       compute_filter = compute,
)[0]) for k,v in to_use.items()}

[f.logger.setLevel(logging.INFO) for f in [crick, boulder,fp]];

# FIGURE DIRECTORIES
SAVEPLOTS = True # Whether to actually make the plots
fig_dir_full        = fpft.get_fig_dir(window_shape = window_shape, window_length = window_length, fit_k = fit_k, create = True); print(f"{fig_dir_full=}")
fig_dir_wnd_shp_len = fpft.get_fig_dir(window_shape = window_shape, window_length = window_length, fit_k = None,  create = True); print(f"{fig_dir_wnd_shp_len=}")
fig_dir_wnd_shp     = fpft.get_fig_dir(window_shape = window_shape, window_length = None,          fit_k = None,  create = True); print(f"{fig_dir_wnd_shp=}")
fig_dir_top         = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = None,  create = True); print(f"{fig_dir_top=}")
fig_dir_fitk        = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = fit_k, create = True); print(f"{fig_dir_fitk=}")
fig_dir_full='figs/hann/1s/fit_k=True'
fig_dir_wnd_shp_len='figs/hann/1s'
fig_dir_wnd_shp='figs/hann'
fig_dir_top='figs'
fig_dir_fitk='figs/fit_k=True'


# FIGURE SHOWING AN EXAMPLE PLUME AND CORRELATIONS
DEFAULT   = "default"
isdefault = lambda x: type(x) is str and x == DEFAULT

su_ds = [k for k,v in to_use.items() if v["sim_name"].startswith("surr")]
surrQ = lambda x: x in su_ds
print(f"Surrogate datasets = {su_ds}.")

which_srcs   = dict_update_from_field({"bw":[-3750, 3750],
                                       "cr":[490000,510000],
                                       "w1":[107300, 114700], "w2":[218300,225700], "w3":[329300,336700],"w4":[440300,447700],
                                       "12T":[498000,502000], "12Ts":[498000,502000],
                                       "16T":[496000,504000], "16Ts":[496000,504000],
},       
                                      su_ds, "bw")        
t_wnd        = dict_update_from_field({"bw":[-4,4]*UNITS.sec,     "cr":[-0.5, 0.5]*UNITS.sec}, su_ds + ["w1","w2","w3","w4","12T","12Ts","16T","16Ts"], "bw")
which_idists = dict_update_from_field({"bw":[0,2,3],              "cr":[0,10,20]},             su_ds + ["w1","w2","w3","w4","12T","12Ts","16T","16Ts"], "bw")
tticks       = dict_update_from_field({"bw":DEFAULT,              "cr":[39.5,40,40.5]},        su_ds + ["w1","w2","w3","w4","12T","12Ts","16T","16Ts"], "bw")
xticks       = dict_update_from_field({"bw":DEFAULT,              "cr":DEFAULT},               su_ds + ["w1","w2","w3","w4","12T","12Ts","16T","16Ts"], "bw")
yticks       = dict_update_from_field({"bw":DEFAULT,              "cr":DEFAULT},               su_ds + ["w1","w2","w3","w4","12T","12Ts","16T","16Ts"], "bw")
snapshot_time = defaultdict(lambda: 40000*UNITS.ms, {"12T":10001*UNITS.ms, "12Ts":40010*UNITS.ms,"16T":10001*UNITS.ms, "16Ts":40010*UNITS.ms})

for k, F in sorted(data.items()):
    if surrQ(k): continue
    ax_plume, ax_traces, ax_corr = fpf.plot_plumes_demo(F, snapshot_time[k], which_srcs[k],
                                                    t_center = (snapshot_time[k].to(UNITS.ms).magnitude//1000)*1000 * UNITS.ms,
                                                    y_lim = (0,5.01) if not surrQ(k) else (-3.01,3.01), y_ticks = [-3,0,3] if surrQ(k) else None,
                                                        data_dir = os.path.join(boulder.data_root, "saved-snapshots") if k == "bw" else None,
                                                    t_wnd = t_wnd[k], dt = 1 * UNITS.sec, which_idists=which_idists[k])
    not isdefault(tticks[k]) and ax_traces[-1].set_xticks(tticks[k])    
    not isdefault(xticks[k]) and ax_plume.set_xticks(xticks[k])
    not isdefault(yticks[k]) and ax_plume.set_yticks(yticks[k])
    if surrQ(k) or k  in ["bw"]: ax_corr.set_xticks(np.arange(5))
    if surrQ(k): [ax_corr.set_ylim(-0.85,1.05), ax_corr.set_ylabel("Correlation",labelpad=-8)]
    fpft.label_axes([ax_plume, ax_traces[0], ax_corr], "ABC", y = [0.99]*3, fontsize=12, fontweight="bold")
    file_name = f"{fig_dir_wnd_shp_len}/plumes_demo_{k}.pdf"
    #ax_corr.set_ylim(-0.5,1.02)
    SAVEPLOTS and (plt.savefig(file_name, bbox_inches='tight'), flush(f"Wrote {file_name}."));
    sys.stdout.flush(); plt.show()
    plt.close()
