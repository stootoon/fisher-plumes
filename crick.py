import os, sys, logging
from os import path as op
from os.path import join as opj
import pickle
import builtins
import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.stats import skew, kurtosis

import utils

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

def list_datasets():
    return list(set(simulations["name"].values))

def print_datasets():
    INFO("Simulation data available for:")
    datasets = list_datasets();
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
        
    for sname, suffs in sorted(series.items()):
        suffs = sorted(suffs)
        if len(suffs)>5:
            INFO("{:<48s}: {} - {} ({:>2d} datasets)".format(sname, suffs[0], suffs[-1], len(suffs)))
        else:
            for suff in suffs:
                INFO(sname + ("_" + suff if len(suff) else suff))
    
print_datasets()
    
def _load_single_simulation(name):
    if name not in simulations["name"].values:
        raise ValueError("Could not find simulation {} in config file.".format(name))
                         
    sim_dir  = simulations[simulations["name"] == name]["root"].values[0]
    probe_dir = op.join(data_root, sim_dir)
    
    with open(op.join(probe_dir, "probe.coords.p"), "rb") as in_file:
        probe_coords = pickle.load(in_file, encoding = "bytes")
    DEBUG("x: {} - {}".format(*utils.fapply([min, max], [p[0] for p in probe_coords])))
    DEBUG("y: {} - {}".format(*utils.fapply([min, max], [p[1] for p in probe_coords])))    
        
    with open(op.join(probe_dir, "probe.t.p"), "rb") as in_file:
        t = pickle.load(in_file, encoding = "bytes")
    DEBUG("t: {} - {}".format(t[0], t[-1]))
    probe_data = np.squeeze(np.load(op.join(probe_dir, "probe.data.npy")))
    DEBUG("probe_data: {}".format(probe_data.shape))
    return {"probe_t":np.array(t), "probe_coords":probe_coords, "probe_data":probe_data}
    
def load(name):
    return CrickSimulationData(name)

class CrickSimulationData:
    def load_probe_data_(self):
        self.probe_coords, self.probe_t, self.data = utils.dd("probe_coords", "probe_t", "probe_data")(_load_single_simulation(self.name))
        
    def __init__(self, name, tol = 0):
        self.tol = tol
        self.name         = name
        self.load_probe_data_()
        self.nx = len(set([p[0] for p in self.probe_coords]))
        self.ny = len(set([p[1] for p in self.probe_coords]))
        self.nz = len(set([p[2] for p in self.probe_coords]))
        self.source     = simulations[simulations["name"] == name]["source"].values[0]
        self.dimensions = simulations[simulations["name"] == name]["dimensions"].values[0]
        self.fields     = simulations[simulations["name"] == name]["fields"].values[0]
        self.fs         = simulations[simulations["name"] == name]["fs"].values[0]
        INFO("Loaded {}".format(name))
        DEBUG("(nx,ny,nz) = ({},{},{})".format(self.nx, self.ny, self.nz))
        px = [p[0] for p in self.probe_coords]
        py = [p[1] for p in self.probe_coords]
        pz = [p[2] for p in self.probe_coords]
        DEBUG(f"x-range: {min(px):8.3g} - {max(px):.3g}")
        DEBUG(f"y-range: {min(py):8.3g} - {max(py):.3g}")
        DEBUG(f"z-range: {min(pz):8.3g} - {max(pz):.3g}")
        self.used_probe_coords = [] # Locations of the used probes 
        

    def nearest_probe(self, x, y, relative_to_source = False):
        return np.argmin([(x + self.source[0]*relative_to_source - p[0])**2 + (y + self.source[1]*relative_to_source - p[1])**2 for p in self.probe_coords])
        
    def coord2str(self, x,y, relative_to_source = True):
        sx = "{:.1f}".format((x - self.source[0]*relative_to_source)*100)
        sy = "{:.1f}".format((y - self.source[1]*relative_to_source)*100)
        if sx != "0.0" and sy != "0.0":
            s = f"x={sx} cm, y={sy} cm"
        elif sx == "0.0" and sy != "0.0":
            s = f"y={sy} cm"
        elif sx != "0.0" and sy == "0.0":
            s = f"x={sx} cm"
        else:
            s = "@source"
        s = s.replace(".0", "")
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
            DEBUG("Loading all coords for {}.".format(self.name))
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
                new_str = names[i] if names is not None else self.coord2str(cx, cy, relative_to_source = use_relative_coords_in_name)
                if new_str in self.coord_strs:
                    WARN(f"Probe at {new_str} already exists, skipping.")
                    continue
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
