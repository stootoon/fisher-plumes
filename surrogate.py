import os, sys, logging
import numpy  as np
from   copy   import deepcopy
from matplotlib import pyplot as plt
from matplotlib import cm

import utils
import fisher_plumes_tools as fpt

from units import UNITS

logging.basicConfig()
logger = logging.getLogger("surrogate")
logger.setLevel(logging.INFO)
INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug


simulations = ["no_info",
               "one_info",
               "two_info",
               "high",
               "spike_and_slab",
               "blue",
               "red",
               "all_equal"]
def list_datasets():
    return simulations

def print_datasets():
    datasets = list_datasets();
    INFO(f"Surrogate simulation data available for: {datasets}")

print_datasets()
        
def _load_single_simulation(name, n_samples = 3001, fs = 50 * UNITS.Hz):
    if name not in simulations:
        raise ValueError(f"Don't know how to create surrogate data {name=}.")

    probe_grid = {"x":np.array([1]).reshape(1,1), "y":np.array([0]).reshape(1,1)}
    DEBUG("x: {} - {}".format(*utils.fapply([np.min, np.max], [probe_grid["x"]])))
    DEBUG("y: {} - {}".format(*utils.fapply([np.min, np.max], [probe_grid["y"]])))    

    T = 1/fs
    t = np.arange(n_samples) * T
    DEBUG(f"t: {t[0]:1.3f}, {t[1]:1.3f} ... {t[-1]:1.3f}")
    DEBUG(f"fs: {fs}")
    probe_data = [] #np.squeeze(np.load(op.join(probe_dir, "probe.data.npy")))
    #DEBUG("probe_data: {}".format(probe_data.shape))
    return {"probe_t":np.array(t), "probe_grid":probe_grid, "probe_data":probe_data}
        
class SurrogateSimulationData:
    def __init__(self, name, units = UNITS.m, pitch_units = UNITS.m, pitch_sym = "ϕ", tol = 0, n_sources = 8, n_samples = 3001, fs = 50 * UNITS.Hz, **kwargs):
        self.tol = tol
        self.name = name
        self.class_name = self.__class__.__name__
        self.units = units
        self.pitch_units = pitch_units
        self.pitch_sym = pitch_sym
        self.pitch = 1 * self.pitch_units

        self.probe_grid = {"x":np.array([10]).reshape(1,1), "y":np.array([0]).reshape(1,1)}
        self.probe_grid["x"]*=self.pitch
        self.probe_grid["y"]*=self.pitch
        self.x  = self.probe_grid["x"][:,0] 
        self.nx = len(self.x)
        self.y  = self.probe_grid["y"][0]
        self.ny = len(self.y)
        self.nz = 1
        self.x_lim      = sorted([self.x[0], self.x[-1]]) 
        self.y_lim      = sorted([self.y[0], self.y[-1]])
        yvals_um = (np.arange(n_sources)  - (n_sources-1)/2) * 7500 # 7500 is the source spacing of the boulder data
        self.source     = np.array([(0,yi) for yi in yvals_um]) * UNITS.um
        self.dimensions = [self.x[-1] - self.x[0], self.y[-1] - self.y[0]]# * self.units
        self.fields     = [f"S{i}" for i in range(n_sources)]
        self.fs         = fs
        self.surr_data_args = {"type":name, "n_samples":n_samples, "n_sources":n_sources, "fs":fs}
        self.surr_data_args.update(kwargs)
        
        INFO(self)

    def generate_surrogate_data(self, seed=0):
        self.data = {}

        fs = self.fs.to(UNITS.Hz).magnitude

        one_over_f = lambda f,k,fc: 1/(max(f/fc,1)**k)

        n_freq      = self.surr_data_args["n_samples"]//2
        surrogate_k = self.surr_data_args["surrogate_k"] if "surrogate_k" in self.surr_data_args else 4.

        self.nt = 2 * n_freq + 1        
        if self.name == "no_info":
            INFO(f"Generating surogate data where all frequencies are uninformative. {surrogate_k=:.1f}")                                    
            ker_freq = lambda i,j,n: one_over_f(n/self.nt*fs, k=surrogate_k, fc = 1)
            kernel   = lambda i,j,n: (i==j) * ker_freq(i,j,n)
        elif self.name == "all_equal":
            INFO(f"Generating surogate data where all frequencies are equally informative.")                        
            ker_freq = lambda i,j,n: one_over_f(n/self.nt*fs, k=surrogate_k, fc = 1)
            ker_spat = lambda i,j,n: 1 - abs(i-j)/5*0.5
            ker_spat = lambda i,j,n: 2*np.exp(-abs(i-j)/12) - 1
            #ker_spat = lambda i,j,n: np.exp(-abs(i-j))
            kernel   = lambda i,j,n: ker_spat(i,j,n) * ker_freq(i,j,n)
        elif self.name == "one_info":
            INFO(f"Generating surogate data where one frequency is more informative than the others.")            
            ker_freq = lambda i,j,n: one_over_f(n/self.nt*fs, k=surrogate_k, fc = 1)
            ker_spat = lambda i,j,n: 2*np.exp(-abs(i-j)/(12 - 4 * (n==4))) - 1
            kernel   = lambda i,j,n: ker_spat(i,j,n) * ker_freq(i,j,n)                        
        elif self.name == "high":
            INFO(f"Generating surogate data where high frequencies are more informative.")
            ker_freq = lambda i,j,n: one_over_f(n/self.nt*fs, k=surrogate_k, fc = 1)            
            ker_spat = lambda i,j,n: 1 if (i==j) else np.exp(-abs(i-j)/(12. if n< n_freq//2 else 2.))
            kernel   = lambda i,j,n: ker_spat(i,j,n) * ker_freq(i,j,n)                        
        else:
            raise NotImplementedError(f"Surrogate data of type {self.name} not implemented.")

        n_src  = len(self.fields)
        K = np.zeros((n_src * n_freq, n_src * n_freq))
        for n in range(n_freq):
            for i in range(n_src):
                for j in range(n_src):
                    K[n_src * n + i, n_src * n + j] = kernel(i,j,n)

        L = np.linalg.cholesky(K)

        np.random.seed(seed)
        # Xr, Xc = np.random.randn(2, n_src*n_freq) @ L.T
        # c = (Xr + Xc)/2
        # s = (Xr - Xc)/2
        c, s = np.random.randn(2, n_src*n_freq) @ L.T
        
        t = np.arange(0,2*n_freq+1)
        f = 2*np.pi*np.arange(1,n_freq+1)/(2*n_freq)
        C = np.cos(np.outer(t, f))
        S = np.sin(np.outer(t, f))
        x = [C @ c[i::n_src] + S @ s[i::n_src] for i in range(n_src)]
        X = np.array(x)

        for fld, Xi in zip(self.fields, X):
            self.data[fld] = [Xi]

        for p in self.data:
            self.data[p] = np.array(self.data[p]).T
            INFO(f"Field {p} has shape {self.data[p].shape}.")

        self.t = (t / fs) * UNITS.sec
        self.nt= len(self.t)
        t = self.t
        INFO(f"Generated surrogated data for {n_src} sources.")
        INFO(f"t-range: {t[0]:.3f}, {t[1]:.3f} ... {t[-1]:.3f} ({self.nt} points)")
        
        
    def __str__(self):
        s = [f"\n{self.name} {self.__class__}"]
        x = self.x
        y = self.y
        # t = self.t
        s.append(f"{1 * self.pitch_units} = {(1 * self.pitch_units).to(UNITS.m)}")
        s.append(f"x_lim: {self.x_lim[0]:1.6g} to {self.x_lim[1]:1.6g}")
        s.append(f"y_lim: {self.y_lim[0]:1.6g} to {self.y_lim[1]:1.6g}")                
        s.append(f"x-y Dimensions: {self.dimensions}")
        s.append(f"x-range: {x[0]:.3f}, ... {x[-1]:.3f} ({self.nx} points)")
        s.append(f"y-range: {y[0]:.3f}, ... {y[-1]:.3f} ({self.ny} points)")
        # Time data won't be created until surrogate data is actually generated
        # s.append(f"t-range: {t[0]:.3f}, {t[1]:.3f} ... {t[-1]:.3f} ({self.nt} points)")
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
            pitch_string = self.pitch_units.__str__().split(" ")[-1]
            s = s.replace(pitch_string, self.pitch_sym)
        s = s.replace(f"x=0 {self.pitch_sym}, ", "")
        s = s.replace(f"x=-0 {self.pitch_sym}, ", "")        
        s = s.replace(f", y=0 {self.pitch_sym}", "")
        s = s.replace(f", y=-0 {self.pitch_sym}", "")                
        return s
            
    def get_key(self):
        return int(self.source[1].to("um").magnitude)

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
        self.generate_surrogate_data()

        self.colors = {c:cm.winter(np.random.rand()) for c in self.coord_strs}
        return self

    def get_used_probe_coords(self):
        unpack_ = lambda ix, iy: (self.x[ix], self.y[iy])
        return [unpack_(*self.coord_inds[cs]) for cs in self.coord_strs]
    
    def cleanup_probe_data(self, x):
        return x*(x > self.tol)

def load_sims(name, which_coords, py_mode = "absolute", pairs_mode = "all", units = UNITS.m, pitch_units = UNITS.m,
              n_sources = 8, n_samples = 3001, fs = 50 * UNITS.Hz, **kwargs):

    INFO(f"load_sims called for {name=} with {pitch_units=}")
    py_mode = fpt.validate_py_mode(py_mode)
    ssd     = SurrogateSimulationData(name, units = units, pitch_units = pitch_units, n_sources = n_sources, n_samples = n_samples, fs = fs, **kwargs)
    ssd.use_coords([(px, py if py_mode == "absolute" else (py * ssd.dimensions[1].magnitude + ssd.y_lim[0])) for (px, py) in which_coords])
    sims = {}
    for i, (k, v) in enumerate(ssd.data.items()):
        # Have to strip the units because quantities with units don't work well as dictionary keys
        k1 = int(ssd.source[i][1].to(UNITS.um).magnitude) # Get the y-value of the source in um        
        sims[k1]        = deepcopy(ssd)
        sims[k1].data   = v.copy()
        sims[k1].fields = [ssd.fields[i]]
        sims[k1].source = ssd.source[i]
    yvals = list(sims.keys())
    pairs_um = fpt.compute_pairs(yvals, pairs_mode)
    return sims, pairs_um
