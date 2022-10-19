import os, sys
import logging
import numpy as np
from copy import deepcopy
from scipy.signal import stft, tukey
from scipy.stats  import kstest
from scipy.optimize import curve_fit, minimize
from matplotlib import pylab as plt

import utils

logger = utils.create_logger("fisher_plumes_tools")
logger.setLevel(logging.DEBUG)

INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

import boulder
import crick

def compute_pairs(yvals, pairs_mode="all"):
    INFO(f"Computing pairs for {len(yvals)=} from {np.min(yvals)} to {np.max(yvals)} using {pairs_mode=}.")
    nyvals = len(yvals)
    pairs = {}
    if pairs_mode == "all":
        for i, y1 in enumerate(yvals):
            for y2 in yvals[i:]: # Start at i instead of i+1 so that we get the dist 0 data as well.
                if (y2-y1) not in pairs:
                    pairs[y2-y1] = [(y2,y1)]
                else:
                    pairs[y2-y1].append((y2,y1))
        
                if (y1-y2) not in pairs:
                    pairs[y1-y2] = [(y1,y2)]
                else:
                    pairs[y1-y2].append((y1,y2))
    elif pairs_mode == "unsigned":
        for i, y1 in enumerate(yvals):
            for y2 in yvals[i:]: # Start at i instead of i+1 so that we get the dist 0 data as well.
                k = np.abs(y2-y1)
                if k not in pairs:
                    pairs[k] = [(y1,y2), (y2,y1)]
                else:
                    pairs[k].append((y1,y2))
                    pairs[k].append((y2,y1))                    
    elif pairs_mode == "sym":
        for i in range(nyvals//2+ 1):
            y1, y2 = i, nyvals - 1 - i
            pairs[y1-y2]= [(y1,y2)]
            pairs[y2-y1]= [(y2,y1)]
        pairs[0] = [(y,y) for y in yvals]
    else:
        raise ValueError(f"Don't know what to do for {pairs_mode=}.")
    INFO("Removing duplicates in pairs dictionary.")
    pairs = {d:list(set(p)) for d,p in pairs.items()} # Remove any duplicates
    return pairs

def validate_py_mode(py_mode):
    if py_mode.lower()[:3] == "abs":
        py_mode = "abs"
        logger.info("Using absolute py coordinates.")
    elif py_mode.lower()[:3] == "rel":
        py_mode = "rel"
        INFO("Using relative py coordinates.")
    else:
        raise ValueError(f"{py_mode=} was not 'absolute' or 'rel'.")
    return py_mode

def validate_coords(which_coords):
    if type(which_coords[0]) is not tuple:
        raise ValueError(f"{type(which_coords[0])=} was not tuple.")
    return which_coords

def load_boulder_16_source_sims(which_coords, py_mode = "absolute", pairs_mode = "all", prefix = 'Re100_0_5mm_50Hz_16source', suffix = 'wideDomain.old'):
    py_mode = validate_py_mode(py_mode)
    file_name = prefix
    if suffix: file_name += "_"+suffix
    file_name += ".h5"
    logger.info(f"Loading data from {file_name=}.")
    bb = boulder.load(file_name)
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
    pairs = compute_pairs(yvals, pairs_mode)
    return sims, pairs

def load_crick(sim_name = "n12dishT", which_coords=[(1,  0.5)], py_mode = "absolute", pairs_mode = "all", extract_plumes = False):
    logger.info(f"load_sims for {sim_name=} with {which_coords=} ({py_mode=}).")

    py_mode      = validate_py_mode(py_mode)
    which_coords = validate_coords(which_coords)

    if type(which_coords[0]) is not tuple:
        raise ValueError(f"{type(which_coords[0])=} was not tuple.")
    
    datasets = [s for s in crick.list_datasets() if "{}_".format(sim_name) in s]

    yvals    = [int(s[-3:]) for s in datasets]

    sims = {}
    for y in yvals:
        k = int(y*1000) # y location in microns
        sims[k] = crick.load("ff_int_sym_slow_high_tres_wide_{}_Y0.{:03d}".format(sim_name, y))
        sims[k].use_coords([(px, py * sims[k].dimensions[1] ** (py_mode == "rel")) for (px,py) in which_coords])

    yvals = sorted(sims)
    logger.info("Yvals: {}".format(yvals))
    logger.info("Computing distance pairings.")
    pairs = compute_pairs(yvals, pairs_mode)
    pair_vals = sorted(pairs)
    logger.info("{} distance pairings found, from {} to {}".format(len(pair_vals), min(pair_vals), max(pair_vals)))
    return sims, pairs

class Detrenders:
    @staticmethod
    def normalizer(x):
        return Detrenders.tukey_normalizer(x, tukey_param=0)
    
    @staticmethod
    def tukey_normalizer(x, tukey_param=0.1):
        wnd = x.shape[1]
        y = x*tukey(wnd,tukey_param)[np.newaxis,:]
        return (y - np.mean(y,axis=-1)[:,np.newaxis])/(np.std(y,axis=-1)[:,np.newaxis]+1e-12)

    @staticmethod
    def windowed(x):
        wnd = x.shape[1]
        y = x*tukey(wnd,0.1)[np.newaxis,:]
        return y

    @staticmethod
    def mean_subtract(x):
        y = x.copy()
        return (y - np.mean(y,axis=-1)[:,np.newaxis])
    
    @staticmethod
    def identity(x):
        return x

def compute_sin_cos_stft(data, istart, wnd, ov, detrender=Detrenders.tukey_normalizer, x_only = False, force_nonnegative=False, window = 'boxcar'):
    block_starts = list(range(istart, len(data)-wnd, wnd-ov))
    x = np.copy(data[block_starts[0]:block_starts[-1]+wnd])
    if force_nonnegative:
        x[x<0] = 0
        
    if x_only:
        return x

    freqs,times, S = stft(x, fs = 1, window=window,
                          nperseg=wnd, noverlap=ov, detrend= detrender,
                          boundary=None, padded=False)
    n =  np.arange(wnd//2+1)
    n = 1 + (n > 0)*(n != wnd//2)
    c =  np.real(S)*n[:, np.newaxis]
    s = -np.imag(S)*2
    return s.T, c.T, times.astype(int)

# CDF of the asymmetric laplacian
alaplace_cdf = lambda la, mu, x: (mu/(la + mu))*np.exp(-2*np.abs(x)/mu)*(x<=0)+ ((mu + la*(1 - np.exp(-2*np.abs(x)/la)))/(mu+la))*(x>0)

# COMPUTE P VALUE FOR THE KOLMOGOROV SMIRNOV TEST
compute_ks_pvalue = lambda la, mu, x: kstest(x, lambda x: alaplace_cdf(la, mu, x)).pvalue

gen_exp     = lambda d, a, s, k, b: (a - b) * np.exp(-np.abs(d/s)**k) + b
fit_gen_exp = lambda d, la: curve_fit(gen_exp, d, la,
                                      p0=[np.max(la),1,1,0],
                                      bounds=(0, np.inf))[0]
fit_gen_exp_no_amp = lambda d, la, a: curve_fit(lambda d, s, k, b: gen_exp(d, a, s, k, b),
                                                d, la,
                                                p0=[1,1,0],
                                                bounds=(0, np.inf))[0]

def compute_fisher_information_for_gen_exp_decay(s, γ, k):
    sn = s/γ
    return (k/γ)**2 * (sn ** (k - 1))**2/(np.exp(sn ** k) - 1)

