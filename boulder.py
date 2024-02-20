import os, sys, logging
from os import path as op
from os.path import join as opj
import json
import h5py
import pickle
import re

import pdb

import pandas as pd
from   copy   import deepcopy
import numpy  as np
from matplotlib import pyplot as plt
from matplotlib import cm

import utils
import fisher_plumes_tools as fpt

from units import UNITS

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
        
class BoulderSimulationData:
    def init_from_dict(self, d, overwrite = False):
        """
        Initialize this object from a dictionary.
        Only copy data fields, not methods.
        """
        INFO(f"Attempting to copy data fields.")
        copied = []
        for k,v in d.items():
            if not callable(v):
                if k in self.__dict__ and not overwrite:
                    DEBUG(f"Skipping field {k} because it already exists.")
                else:
                    self.__dict__[k] = deepcopy(v)
                    copied.append(k)
                    DEBUG(f"Copied field {k}.")
        return copied
    
    def __init__(self, name, units = UNITS.m, pitch_units = UNITS.m, pitch_sym = "ϕ", tol = 0):
        if type(name) is dict:
            INFO(f"Initializing from dictionary.")
            self.init_from_dict(name)
        else:
            self.tol = tol
            self.name = name
            self.class_name = self.__class__.__name__
            self.probe_grid, self.probe_t, self.data = utils.dd("probe_grid", "probe_t", "probe_data")(_load_single_simulation(name))
            self.units = units
            self.pitch_units = pitch_units
            self.pitch_sym = pitch_sym
            self.probe_t *= UNITS.s
            self.t  = self.probe_t
            self.nt = len(self.probe_t)
            self.probe_grid["x"]*=self.units
            self.probe_grid["y"]*=self.units
            self.x  = self.probe_grid["x"][:,0] 
            self.nx = len(self.x)
            self.y  = self.probe_grid["y"][0]
            self.ny = len(self.y)
            self.nz = 1
            self.x_lim      = sorted([self.x[0], self.x[-1]]) 
            self.y_lim      = sorted([self.y[0], self.y[-1]]) 
            self.source     = simulations[simulations["name"] == name]["source"].values[0] * self.units
            self.dimensions = simulations[simulations["name"] == name]["dimensions"].values[0] * self.units
            self.fields     = simulations[simulations["name"] == name]["data_fields"].values[0]
            self.fs         = simulations[simulations["name"] == name]["fs"].values[0] * UNITS.Hz
            self.integral_length_scales = {}
            INFO(self)
    
    def __str__(self):
        s = [f"\n{self.name} <BoulderSimulationData>"]
        x = self.x
        y = self.y
        t = self.t
        s.append(f"x_lim: {self.x_lim[0]:1.6g} to {self.x_lim[1]:1.6g}")
        s.append(f"y_lim: {self.y_lim[0]:1.6g} to {self.y_lim[1]:1.6g}")                
        s.append(f"x-y Dimensions: {self.dimensions}")
        s.append(f"x-range: {x[0]:.3f}, {x[1]:.3f} ... {x[-1]:.3f} ({self.nx} points)")
        s.append(f"y-range: {y[0]:.3f}, {y[1]:.3f} ... {y[-1]:.3f} ({self.ny} points)")
        s.append(f"t-range: {t[0]:.3f}, {t[1]:.3f} ... {t[-1]:.3f} ({self.nt} points)")
        s.append(f"fs: {self.fs:g}")
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

    def coord2str(self, x, y, output_units = None):
        if output_units is None: output_units = self.pitch_units
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
        return int(self.source[1].to("um").magnitude)
    
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
                raise ValueError(f"Number of names {len(names)} does not match number of coords {len(coords)}.")

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

    def compute_integral_length_scale(self, which_dir, px, py, recompute = False):
        assert which_dir in ["x", "y"], f"which_dir must be 'x' or 'y', not '{which_dir}'."
        INFO(f"Computing integral length scale at {px=}, {py=} for {which_dir=}.")                
        k = (px,py,which_dir) 
        if k in self.integral_length_scales and recompute == False:
            INFO(f"Using cached value for {k=}.")
            v = self.integral_length_scales[k]
            return v["l"], v["fr"]

        ix, iy = self.nearest_probe(px, py)
        INFO(f"Nearest probe index is {(ix, iy)}.")
        with self.load_h5() as f:
            if which_dir == "x":
                s  = f["Flow Data"]["u"][:, :, iy]
                ii = ix
                ds = abs(self.probe_grid["x"][1,0] - self.probe_grid["x"][0,0])
            elif which_dir == "y":
                s  = f["Flow Data"]["v"][:, ix, :]
                ii = iy
                ds = abs(self.probe_grid["y"][0,1] - self.probe_grid["y"][0,0])
            else:
                raise ValueError(f"Unknown direction '{which_dir}'.")

        sms = s - np.mean(s, axis=0)
        ss  = {}
        for r in range(sms.shape[1]):
            ss[r] = []
            if ii - r >=0:            ss[r].append(np.mean(sms[:,ii-r]*sms[:,ii]))
            if ii + r < sms.shape[1]: ss[r].append(np.mean(sms[:,ii+r]*sms[:,ii]))
        fr = np.array([np.mean(ssr) if len(ssr) else np.nan for _,ssr in sorted(ss.items())])
        fr/= fr[0]
        l  = np.nansum(fr * ds)

        self.integral_length_scales[(px, py, which_dir)] = {"fr":fr, "l":l, "ds":ds}

        return l, fr, ds

def load_sims(which_coords, py_mode = "absolute", pairs_mode = "all",
              units = UNITS.m, pitch_units = UNITS.m,
              prefix = 'Re100_0_5mm_50Hz_16source', suffix = 'wideDomain.orig',
              pair_resolution_um = 100,
):
    py_mode = fpt.validate_py_mode(py_mode)
    file_name = prefix
    if suffix: file_name += "_"+suffix
    file_name += ".h5"
    logger.info(f"Loading data from {file_name=}.")
    bb = BoulderSimulationData(file_name,units=units, pitch_units = pitch_units)
    fixed_sources = []

    for x,y in bb.source:
        if np.abs(y.magnitude)>0.3:
            WARN(f"Found incorrect source {y=}.")
            y/=10
            WARN(f"Corrected to {y=}.")            
        fixed_sources.append([x.magnitude,y.magnitude])
    if "old" in file_name:
        WARN("Doubling y coordinates because they were wrong in the original data.")
        fixed_sources = [(x.magnitude,2*y.magnitude) for x,y in fixed_sources]

    bb.source = np.array(fixed_sources) * bb.units

    bb.use_coords([(px, py if py_mode == "absolute" else (py * bb.dimensions[1].magnitude + bb.y_lim[0])) for (px, py) in which_coords])
    sims = {}

    # Get the x and y coordinates of all the sources
    xvals = np.array([x.to(UNITS.um).magnitude for x, _ in bb.source])
    yvals = np.array([y.to(UNITS.um).magnitude for _, y in bb.source])
    svals, source_line = fpt.compute_source_line(xvals, yvals, "mean", )

    for i, (k, v) in enumerate(bb.data.items()):
        # Have to strip the units because quantities with units don't work well as dictionary keys
        k1 = int(svals[i]) # Get the y-value of the source in um        
        sims[k1] = deepcopy(bb)
        sims[k1].data = v.copy()
        sims[k1].fields = [bb.fields[i]]
        sims[k1].source = bb.source[i]
    yvals = list(sims.keys())
    pairs_um = fpt.compute_pairs(yvals, pairs_mode, pair_resolution_um)
    return sims, pairs_um, source_line

