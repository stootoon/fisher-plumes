import os, sys, logging
from os import path as op
from os.path import join as opj
import pickle
import json

import pandas as pd
import numpy as np
from matplotlib import cm

import imageio as iio # For importing field snapshot PNGs

from scipy.stats import skew, kurtosis

import utils
import fisher_plumes_tools as fpt

import pdb

from units import UNITS

logging.basicConfig()
logger = logging.getLogger("crick")
logger.setLevel(logging.INFO)
INFO = logger.info
DEBUG= logger.debug
WARN = logger.warning

curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(curr_dir, "crick.json"), "r") as in_file:
    config = json.load(in_file)

data_root   = utils.expand_environment_variables(config["root"])

simulations = pd.DataFrame(config["registry"])

def list_datasets(as_series = False):
    datasets = list(set(simulations["name"].values))
    if as_series:
        series = {}
        for ds in datasets:
            if "_" in ds:
                dss = ds.split("_")
                series_name = "_".join(dss[:-1])
                suffix      = dss[-1]
            else:
                series_name = ds
                suffix = ""
            if series_name in series:
                series[series_name].append(suffix)
            else:
                series[series_name] = [suffix]
        return series
    else:
        return datasets
        

def print_datasets():
    INFO("Simulation data available for:")
    series = list_datasets(as_series = True);
        
    for sname, suffs in sorted(series.items()):
        suffs = sorted(suffs)
        if len(suffs)>5:
            INFO("{:<48s}: {} - {} ({:>2d} datasets)".format(sname, suffs[0], suffs[-1], len(suffs)))
        else:
            for suff in suffs:
                INFO(sname + ("_" + suff if len(suff) else suff))
    
print_datasets()
    
def _load_single_simulation(name, max_time = np.inf * UNITS.sec):
    if name not in simulations["name"].values:
        raise ValueError("Could not find simulation {} in config file.".format(name))
                         
    sim_dir  = simulations[simulations["name"] == name]["root"].values[0]
    probe_dir = op.join(data_root, sim_dir)

    file_name = op.join(probe_dir, "probe.coords.p")
    INFO(f"Reading probe COORDINATES from {file_name=}.")
    with open(file_name, "rb") as in_file:
        probe_coords = pickle.load(in_file, encoding = "bytes")
    x, y = zip(*[(p[0], p[1]) for p in probe_coords])
    x = np.array(x)
    y = np.array(y)
    DEBUG(f"x: {x.min()} - {x.max()} {len(np.unique(x))=}")
    DEBUG(f"y: {y.min()} - {y.max()} {len(np.unique(y))=}")

    file_name = op.join(probe_dir, "probe.t.p")
    INFO(f"Reading probe TIMES       from {file_name=}.")    
    with open(file_name, "rb") as in_file:
        t = np.array(pickle.load(in_file, encoding = "bytes"))
    DEBUG(f"{t[0]=} - {t[-1]=} {len(t)=}")
    max_time_sec = max_time.to(UNITS.s).magnitude
    ind_keep = t <= max_time_sec
    DEBUG(f"Keeping first {sum(ind_keep)} of {len(ind_keep)} time points.")

    file_name = op.join(probe_dir, "probe.data.npy")
    INFO(f"Reading probe DATA        from {file_name}.")
    probe_data = np.squeeze(np.load(file_name))
    DEBUG("probe_data: {}".format(probe_data.shape))
    n_t_avail = probe_data.shape[0]    
    if n_t_avail < len(ind_keep):
        WARN(f"Probe data had {n_t_avail=} timepoints < {len(ind_keep)=}.")
    probe_data = probe_data[ind_keep[:n_t_avail]]
    t = t[ind_keep][:n_t_avail]
    return {"probe_t":np.array(t), "probe_coords":probe_coords, "probe_data":probe_data}
    
class CrickSimulationData:
    def load_probe_data_(self, max_time = np.inf * UNITS.s):
        self.probe_coords, self.probe_t, self.data = utils.dd("probe_coords", "probe_t", "probe_data")(_load_single_simulation(self.name, max_time = max_time))
        INFO(f"Loaded probe data. {len(self.probe_coords)=} {len(self.probe_t)=} {self.data.shape=}.")
        assert len(self.probe_t)      == self.data.shape[0], f"{len(self.t)=} <> {self.data.shape[0]=}"                
        assert len(self.probe_coords) == self.data.shape[1], f"{len(self.probe_coords)=} <> {self.data.shape[1]=}"

        
    def __init__(self, name, units = UNITS.m, pitch_units = UNITS.m, pitch_sym = "ϕ", tol = 0, max_time = np.inf * UNITS.s, snapshots_folder = None):
        self.tol  = tol
        self.name = name
        self.load_probe_data_(max_time = max_time)
        self.units = units
        self.pitch_units = pitch_units
        self.pitch_sym = pitch_sym
        self.probe_t *= UNITS.s
        self.t = self.probe_t
        self.x, self.y, self.z = [np.array(u) * self.units for u in zip(*self.probe_coords)]
        self.probe_coords *= self.units        
        self.nx, self.ny, self.nz = [len(set(u)) for u in [self.x, self.y, self.z]]
        self.source     = simulations[simulations["name"] == name]["source"].values[0] * self.units
        self.dimensions = simulations[simulations["name"] == name]["dimensions"].values[0] * self.units
        self.fields     = simulations[simulations["name"] == name]["fields"].values[0] 
        self.fs         = simulations[simulations["name"] == name]["fs"].values[0] * UNITS.Hz
        self.x_lim = [0 * self.units, self.dimensions[0]]
        self.y_lim = [0 * self.units, self.dimensions[1]]
        INFO(f"Loaded {name} with data at {len(self.probe_coords)} locations.")
        INFO(f"1 {pitch_sym} = {(1 * pitch_units).to(units)}")
        DEBUG(f"(nx,ny,nz) = ({self.nx},{self.ny},{self.nz})")
        DEBUG(f"x-range: {min(self.x):8.3g} - {max(self.x):.3g}")
        DEBUG(f"y-range: {min(self.y):8.3g} - {max(self.y):.3g}")
        DEBUG(f"z-range: {min(self.z):8.3g} - {max(self.z):.3g}")        
        self.used_probe_coords = [] # Locations of the used probes
        if snapshots_folder is not None: self.init_snapshots(snapshots_folder)

    def init_snapshots(self, root_folder, snapshot_finder_fun = None):
        INFO(f"Initializing snapshots folder to {root_folder}.")
        if not os.path.exists(root_folder): raise FileExistsError(f"Snapshots folder {root_folder} not found.")
        if snapshot_finder_fun: self.snapshot_finder_fun = snapshot_finder_fun
        else: self.snapshot_finder_fun = lambda fld, dim, time: os.path.join(root_folder, f"{fld}_d{dim}_{int(time.to(UNITS.ms).magnitude):06}.png")

    def get_snapshot(self, fld, time, which_dim = 0, which_channel = 0, normalizer = 255.):
        if not hasattr(self, "snapshot_finder_fun"): raise AttributeError("Missing 'snapshot_finder_fun'. Run init_snapshots first.")
        file_name = self.snapshot_finder_fun(fld, which_dim, time)
        if not os.path.exists(file_name): raise FileExistsError(f"Could not find {file_name=}.")
        img = iio.imread(file_name)
        return np.array(img[:,:,which_channel])/normalizer    
            
    def nearest_probe(self, x, y, relative_to_source = False):
        xx = x + self.source[0] * float(relative_to_source)
        yy = y + self.source[1] * float(relative_to_source)
        dx = (xx - self.x)**2
        dy = (yy - self.y)**2
        dd = dx + dy
        return np.argmin(dd)
        
    def coord2str(self, x,y, relative_to_source = True, output_units = None):
        if output_units is None: output_units = self.pitch_units
        if relative_to_source:
            x -= self.source[0]
            y -= self.source[1]
        sx = f"{x.to(output_units):.1f}"
        sy = f"{y.to(output_units):.1f}"
        if x != 0 and y != 0:
            s = f"x={sx}, y={sy}"
        elif sx == 0 and sy != 0:
            s = f"y={sy}"
        elif sx != 0 and sy == 0:
            s = f"x={sx}"
        else:
            s = "@origin"
        s = s.replace(".0", "")
        if output_units == self.pitch_units:
            pitch_string = self.pitch_units.__str__().split(" ")[1]
            s = s.replace(pitch_string, self.pitch_sym)
        s = s.replace(f"x=0 {self.pitch_sym}, ", "")
        s = s.replace(f"x=-0 {self.pitch_sym}, ", "")        
        s = s.replace(f", y=0 {self.pitch_sym}", "")
        s = s.replace(f", y=-0 {self.pitch_sym}", "")        
        return s
        
    def use_coords(self, coords, use_relative_coords_in_name = True, names = None):
        """ Which of the probe coords to actually use. 
        The values should be in absolute coordinates.
        """
        if names is not None:
            if len(names) != len(coords):
                raise ValueError("Number of names {} does not match number of coords {}.".format(len(names), len(coords)))

        if len(self.used_probe_coords):
            # Reload all the data, so we can subset it below
            self.load_probe_data_()
            
        self.coord_strs = [] # The coordinates as strings
        self.coord_inds = {} # The indices, indexed by coordinate strings, into the probe data
            
        if coords == "all":
            DEBUG(f"Loading all coords for {self.name}.")
            coords = range(len(self.probe_coords))

        if type(coords[0]) is not tuple:
            # Here the actual indices of the coords we want to use are provided.
            # So use those.
            data_inds = []
            self.used_probe_coords = []
            for i, icoord in enumerate(coords):
                cx, cy, _ = self.probe_coords[icoord]
                self.coord_strs.append(names[i] if names is not None else self.coord2str(cx, cy, relative_to_source = use_relative_coords_in_name))            
                self.coord_inds[self.coord_strs[-1]] = i
                data_inds.append(icoord)
                self.used_probe_coords.append((cx,cy))
                INFO("Using coordinate index {:>4d} ({:>8.3f}, {:>8.3f}), name {}.".format(icoord, cx, cy,self.coord_strs[-1]))
            self.data = self.data[:,data_inds]

        else:
            data_inds = []
            ic = 0
            self.used_probe_coords = []            
            for i, (x, y) in enumerate(coords):
                inearest = self.nearest_probe(x, y)
                cx, cy, _ = self.probe_coords[inearest]
                coord_str = names[i] if names is not None else self.coord2str(cx, cy, relative_to_source = use_relative_coords_in_name)
                copies  = 0
                new_str = coord_str
                while new_str in self.coord_strs:
                    WARN(f"Probe with name {new_str} already exists, renaming.")
                    copies += 1
                    new_str = coord_str + f"copy {copies}"
                    
                self.coord_strs.append(new_str)            
                self.coord_inds[self.coord_strs[-1]] = ic
                ic +=1
                data_inds.append(inearest)
                self.used_probe_coords.append((cx,cy))                
                INFO("Mapped coordinate ({:6.3f}, {:6.3f}) to ({:6.3f}, {:6.3f}), index {}, name {}.".format(x,y, cx, cy, inearest, self.coord_strs[-1]))
            self.data = self.data[:,data_inds]                
            
        self.colors = {c:cm.winter(np.random.rand()) for c in self.coord_strs}
        self.t = np.arange(self.data.shape[0])/self.fs
        return self
    
    def cleanup_probe_data(self, x):
        return x*(x > self.tol)

    def get_used_probe_coords(self):
        return self.used_probe_coords
    

    def get_key(self):
        return int(self.source[1].to("um").magnitude)


def load_sims(sim_name = "n12dishT", which_coords=[(1,  0.5)], max_time = np.inf * UNITS.s, units = UNITS.m, pitch_units = UNITS.m, py_mode = "absolute", pairs_mode = "all", extract_plumes = False):
    INFO(f"load_sims for {sim_name=} with {which_coords=} ({py_mode=}).")

    py_mode      = fpt.validate_py_mode(py_mode)

    datasets = [s for s in list_datasets() if f"{sim_name}_" in s]
    sims = {}
    for ds in datasets:
        INFO("*"*100)
        INFO(f"Loading dataset {ds}.")
        snapshots_folder = None
        if sim_name.startswith("crimgrid"): snapshots_folder = os.path.join(os.getenv("FISHER_PLUMES_DATA"), "crick", ds, "png")
        new_sim = CrickSimulationData(ds, units=units, pitch_units = pitch_units, max_time = max_time, snapshots_folder = snapshots_folder)
        new_yval_mm = new_sim.source[1].to(UNITS.mm).magnitude
        k = int(new_yval_mm*1000) # y location in microns
        sims[k] = new_sim
        sims[k].use_coords([(px, py * (sims[k].dimensions[1].magnitude ** (py_mode == "rel"))) for (px,py) in which_coords])
        ind_nan = np.isnan(sims[k].data)
        n_nan = np.sum(ind_nan)
        if n_nan:
            WARN(f"Found {n_nan=} NaN values. Setting to zero.")
            sims[k].data[ind_nan] = 0
        
    yvals = sorted(list(sims.keys()))

    INFO(f"Yvals: {yvals}")
    INFO(f"Computing distance pairings.")
    pairs = fpt.compute_pairs(yvals, pairs_mode)
    pair_vals = sorted(pairs)
    INFO(f"{len(pair_vals)} distance pairings found, from {min(pair_vals)} to {max(pair_vals)}")
    return sims, pairs
    
