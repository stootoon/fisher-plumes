import os, sys, logging
from os import path as op
from os.path import join as opj
import json
import h5py
import pickle
import re

import pandas as pd
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import utils
import fisher_plumes_tools as fpt

logging.basicConfig()
logger = logging.getLogger("boulder")
logger.setLevel(logging.INFO)
INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(curr_dir, "boulder.json"), "r") as in_file:
    config = json.load(in_file)

data_root   = utils.expand_environment_variables(config["root"])

def parse_registry(registry):
    # If the registry item's 'name' field ends in h5, 
    # loads the metadata from the "meta_data" field,
    # otherwise just passes it through
    parsed = []
    for item in registry:
        # Set some of the fields for hdf5 data from the filename
        pat = "(?P<Re>Re[0-9]+_)(?P<spacing>s[0-9|_]+p_)?(?P<pitch_mm>[0-9|_]+mm_)?(?P<fs>[0-9]+Hz)"
        m = re.search(pat, item["name"])
        item["Re"] = int(m["Re"][2:-1])
        item["fs"] = int(m["fs"][:-2])
        item["pitch_mm"]   = float(m["pitch_mm"][:-3].replace("_","."))
        if m["spacing"] is not None:
            item["spacing_in_pitch"] = float(m["spacing"][1:-2].replace("_","."))
            item["spacing"] = item["spacing_in_pitch"]*item["pitch_mm"]/1000

        # Set some others from the meta data
        data = h5py.File(opj(data_root, item["root"], item["name"]), "r")            
        item["Sc"]      = data["Model Metadata"]["Schmidt number"][0,0]
        item["x_lim"]   = data["Model Metadata"]["xGrid"][[0,-1],0]
        item["y_lim"]   = data["Model Metadata"]["yGrid"][0,[0,-1]]
        item["dimensions"] = np.abs(np.array([np.diff(item["x_lim"]), np.diff(item["y_lim"])]).flatten())
        
        # allow for multiple sources 
        item["source"] = np.array(data["Odor Data"]["odorSourceLocations"]).flatten().reshape(-1,2)
        item["odor_release_speed"] = np.array(data["Odor Data"]["odorReleaseSpeed"][0])
        item["data_fields"] = ["Odor Data/" + fld for fld in data["Odor Data"] if (fld.startswith("c") and fld[1] in "0123456789")]
        parsed.append(item)

    return parsed

registry    = parse_registry(config["registry"])
simulations = pd.DataFrame(registry)

def list_datasets():
    return list(set(simulations["name"].values))

def print_datasets():
    INFO("Boulder simulation data available for:")
    datasets = list_datasets();
    series = {}
    for ds in datasets:
        series[ds] = simulations[simulations.name==ds]["data_fields"].values[0]
        
    for sname, suffs in sorted(series.items()):
        INFO(f"{sname:<48s}: {suffs} ({len(suffs):>2d} datasets)")

print_datasets()
        
def describe_h5(h5, n=0):
    if "keys" in dir(h5):
        for k in h5.keys():
            print("  "*n + k)
            describe_h5(h5[k], n+1)

def concs2rgb(r,b,
              RFUN = lambda r,b: 1-b,
              GFUN = lambda r,b: 1-(r+b),
              BFUN = lambda r,b: 1-r):
    r = np.clip(r,0,1)
    b = np.clip(b,0,1)
    R = RFUN(r,b)    
    G = GFUN(r,b)
    B = BFUN(r,b)
    return np.dstack([R,G,B])

def demo_colormap():
    ll = np.linspace(0,1,1001)
    rr,bb = np.meshgrid(ll,ll)
    zz = (rr+bb)>1
    rr[zz] = 0
    bb[zz] = 0
    plt.imshow(concs2rgb(np.fliplr(rr), np.fliplr(bb)))

def _load_single_simulation(name):
    if name not in simulations["name"].values:
        raise ValueError("Could not find simulation {} in config file.".format(name))

    sim_dir  = simulations[simulations["name"] == name]["root"].values[0]
    probe_dir = op.join(data_root, sim_dir)

    data = h5py.File(opj(probe_dir, name), "r")

    probe_grid = {"x":np.array(data["Model Metadata"]["xGrid"]), "y":np.array(data["Model Metadata"]["yGrid"])}
    DEBUG("x: {} - {}".format(*utils.fapply([np.min, np.max], [probe_grid["x"]])))
    DEBUG("y: {} - {}".format(*utils.fapply([np.min, np.max], [probe_grid["y"]])))    

    t = np.array(data["Model Metadata"]["timeArray"]).flatten()    
    DEBUG(f"t: {t[0]:1.3f}, {t[1]:1.3f} ... {t[-1]:1.3f}")
    DEBUG(f"fs: {1/(t[1]-t[0]):1.2f} Hz")
    probe_data = [] #np.squeeze(np.load(op.join(probe_dir, "probe.data.npy")))
    #DEBUG("probe_data: {}".format(probe_data.shape))
    return {"probe_t":np.array(t), "probe_grid":probe_grid, "probe_data":probe_data}
        
def load(name):
    return BoulderSimulationData(name)

class BoulderSimulationData:
    def __init__(self, name, tol = 0):
        self.tol = tol
        self.name = name
        self.probe_grid, self.probe_t, self.data = utils.dd("probe_grid", "probe_t", "probe_data")(_load_single_simulation(name))
        self.t  = self.probe_t
        self.nt = len(self.probe_t)
        self.x  = self.probe_grid["x"][:,0]
        self.nx = len(self.x)
        self.y  = self.probe_grid["y"][0]
        self.ny = len(self.y)
        self.nz = 1
        self.x_lim      = sorted([self.x[0], self.x[-1]])
        self.y_lim      = sorted([self.y[0], self.y[-1]])
        self.source     = simulations[simulations["name"] == name]["source"].values[0]
        self.dimensions = simulations[simulations["name"] == name]["dimensions"].values[0]
        self.fields     = simulations[simulations["name"] == name]["data_fields"].values[0]
        self.fs         = simulations[simulations["name"] == name]["fs"].values[0]
        INFO(self)

    def __str__(self):
        s = [f"\n{self.name} <BoulderSimulationData>"]
        x = self.x
        y = self.y
        t = self.t
        s.append(f"x_lim: {self.x_lim}")
        s.append(f"y_lim: {self.y_lim}")                
        s.append(f"x-y Dimensions: {self.dimensions}")
        s.append(f"x-range: {x[0]:.3f}, {x[1]:.3f} ... {x[-1]:.3f} ({self.nx} points)")
        s.append(f"y-range: {y[0]:.3f}, {y[1]:.3f} ... {y[-1]:.3f} ({self.ny} points)")
        s.append(f"t-range: {t[0]:.3f}, {t[1]:.3f} ... {t[-1]:.3f} ({self.nt} points)")
        s.append(f"fs: {self.fs:g} Hz")
        s.append("Sources:")
        for i, (fld, src) in enumerate(zip(self.fields, self.source)):
            fld_name = fld.split("/")[-1]
            s.append((f" {i:>2d}: = {fld_name:>6s} @ (x = {src[0]:+.6g}, y = {src[1]:+.6g})"))

        if "coord_strs" in self.__dict__:
            s.append("Probes:")
            for i, cstr in enumerate(self.coord_strs):
                inds = self.coord_inds[cstr]
                s.append(f" {i}: '{cstr}' @ (ix = {inds[0]}, iy = {inds[1]})")
        return "\n".join(s)
        
    def nearest_probe(self, x, y):
        px = self.probe_grid["x"]
        py = self.probe_grid["y"]
        dx = (x - px)**2
        dy = (y - py)**2
        dd = dx + dy
        imin = np.where(dd == np.min(dd))
        ix, iy = imin[0][0], imin[1][0]
        return ix, iy
        
    def coord2str(self, x,y):
        sx = "{:.1f}".format(x*100)
        sy = "{:.1f}".format(y*100)
        if sx != "0.0" and sy != "0.0":
            s = f"x={sx} cm, y={sy} cm"
        elif sx == "0.0" and sy != "0.0":
            s = f"y={sy} cm"
        elif sx != "0.0" and sy == "0.0":
            s = f"x={sx} cm"
        else:
            s = "@origin"
        s = s.replace(".0", "")
        return s

    def load_h5(self):
        sim_dir  = simulations[simulations["name"] == self.name]["root"].values[0]
        probe_dir = op.join(data_root, sim_dir)
        file_path = opj(probe_dir, self.name)
        DEBUG(f"Loading data from {file_path}.")
        return h5py.File(file_path, "r")
    
    def snapshot(self, which_field, which_time, is_index = False):
        INFO(f"Snapshoting {which_field=} at {which_time=}.")
        h5 = self.load_h5()
        for fld in self.fields:
            DEBUG(f"Checking {fld}")
            if fld.endswith(which_field):
                DEBUG(f"Loading data for {which_field=} from h5.{fld}.")
                sub = h5
                for p in fld.split("/"):
                    sub = sub[p]
                sub = np.array(sub)
                expected_shape = (self.nt, self.nx, self.ny) 
                if sub.shape != expected_shape:
                    raise ValueError(f"Data for {which_field=} has shape {sub.shape} != expected_shape")
                index = which_time if is_index else np.argmin(np.abs(self.probe_t - which_time))
                t_index = self.probe_t[index]
                DEBUG(f"Returning snapshot at {t_index:g} sec. ({index=}).")
                return sub[index]
        raise ValueError(f"No field found matching '{which_field}'.")

    def get_key(self):
        return int(self.source[1]*1000000)
    
    def save_snapshot(self, t, data_dir = "."):
        fld = self.fields[0][-3:]
        fld_data  = self.snapshot(fld, t)
        file_name = os.path.join(data_dir, f"{fld}_t{t:g}.p")
        with open(file_name,"wb") as f:
            pickle.dump(fld_data, f)
            INFO(f"Wrote data for {fld} @ {t=} to {file_name}")
                    
    def load_saved_snapshot(self, t, data_dir = "."):
        fld = self.fields[0][-3:]
        file_name = f"{fld}_t{t:g}.p"
        full_file = os.path.join(data_dir, file_name)
        INFO(f"Loading {fld=} at {t=:g} from {full_file=}.")
        return np.load(os.path.join(data_dir, file_name), allow_pickle=True).T

    def use_coords(self, coords, names = None, skip_if_exists = True):
        """ Which of the probe coords to actually use. 
        The values should be in absolute coordinates.
        """
        self.coord_strs = [] # The coordinates as strings
        self.coord_inds = {} # The indices, indexed by coordinate strings, into the probe data

        if names is not None:
            if len(names) != len(coords):
                raise ValueError("Number of names {} does not match number of coords {}.".format(len(names), len(coords)))

        indices = [self.nearest_probe(*xy) for xy in coords]

        for i, (ix, iy) in enumerate(indices):
            cx, cy = self.probe_grid["x"][ix,iy], self.probe_grid["y"][ix,iy]
            probe_name = names[i] if names is not None else self.coord2str(cx, cy)
            if probe_name in self.coord_strs and skip_if_exists:
                logger.warning(f"Probe {probe_name} exists, skipping.")
                continue
            self.coord_strs.append(probe_name)            
            self.coord_inds[self.coord_strs[-1]] = (ix, iy)
            INFO(f"Mapped coordinate ({coords[i][0]:6.3f}, {coords[i][1]:6.3f}) to ({cx:6.3f}, {cy:6.3f}), index {(ix,iy)}, name '{self.coord_strs[-1]}'.")

        # Now actually load the data
        # data = self.load_h5()
        self.data = {}
        for fld in self.fields:
            DEBUG(f"Loading {fld}")
            parts = fld.split("/")

            with self.load_h5() as f:
                # h5_file = self.load_h5() # data # Load it each time as otherwise it seems to keep everything we touched in RAM.
                for i, p in enumerate(parts):
                    DEBUG(" "*(i+1) + f"Getting '{p}'")
                    f = f[p]

                DEBUG(f"data[{fld}] has shape {f.shape}")
                self.data[p] = [np.array(f[:, ix, iy]) for (ix, iy) in [self.coord_inds[s] for s in self.coord_strs]]                        

        for p in self.data:
            self.data[p] = np.array(self.data[p]).T
            INFO(f"Field {p} has shape {self.data[p].shape}.")
            
        self.colors = {c:cm.winter(np.random.rand()) for c in self.coord_strs}
        return self

    def get_used_probe_coords(self):
        unpack_ = lambda ix, iy: (self.x[ix], self.y[iy])
        return [unpack_(*self.coord_inds[cs]) for cs in self.coord_strs]
    
    def cleanup_probe_data(self, x):
        return x*(x > self.tol)

def load_sims(which_coords, py_mode = "absolute", pairs_mode = "all", prefix = 'Re100_0_5mm_50Hz_16source', suffix = 'wideDomain.old'):
    py_mode = fpt.validate_py_mode(py_mode)
    file_name = prefix
    if suffix: file_name += "_"+suffix
    file_name += ".h5"
    logger.info(f"Loading data from {file_name=}.")
    bb = load(file_name)
    fixed_sources = []
    for x,y in bb.source:
        if np.abs(y)>0.3:
            WARN(f"Found incorrect source {y=}.")
            y/=10
            WARN(f"Corrected to {y=}.")            
        fixed_sources.append([x,y])
    if "old" in file_name:
        WARN("Doubling y coordinates because they were wrong in the original data.")
        fixed_sources = [(x,2*y) for x,y in fixed_sources]
        
    bb.source = np.array(fixed_sources)
    bb.use_coords([(px, py if py_mode == "absolute" else (py * bb.dimensions[1] + bb.y_lim[0])) for (px, py) in which_coords])
    sims = {}
    for i, (k, v) in enumerate(bb.data.items()):
        # k = int(("-" if k[-1] == "a" else "")+k[1]) # Names are c2a, c3b etc. so convert to yvals -8 to 8
        k1 = int(bb.source[i][1] * 1000000) # Get the y-value of the source in um
        sims[k1] = deepcopy(bb)
        sims[k1].data = v.copy()
        sims[k1].fields = [bb.fields[i]]
        sims[k1].source = bb.source[i]
    yvals = list(sims.keys())
    pairs = fpt.compute_pairs(yvals, pairs_mode)
    return sims, pairs
