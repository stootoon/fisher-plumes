import os, sys
import logging
import numpy as np
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

# COMPUTE R2 VALUE COMPARING THE PREDICTED AND FIT CDFs
cdfvals = lambda x: np.arange(1,len(x)+1)/len(x)
compute_r2_value  = lambda la, mu, x: 1 - np.mean((cdfvals(x) -  alaplace_cdf(la, mu, np.sort(x)))**2)/np.var(cdfvals(x))

gen_exp     = lambda d, a, s, k, b: (a - b) * np.exp(-np.abs(d/s)**k) + b
fit_gen_exp = lambda d, la: curve_fit(gen_exp, d, la,
                                      p0=[np.max(la),1,1,0],
                                      bounds=(0, np.inf))[0]
fit_gen_exp_no_amp = lambda d, la, a: curve_fit(lambda d, s, k, b: gen_exp(d, a, s, k, b),
                                                d, la,
                                                p0=[1,1,0],
                                                bounds=(0, np.inf))[0]

def compute_fisher_information_for_gen_exp_decay(s, γ, k, b, σ2):
    sn    = s/γ
    c0    = (2*σ2 - b)
    E     = np.exp(-sn**k)

    coef  = 2 * k**2 / γ**2 
    num1  = c0 * (sn**(k-1) * E)**2
    den1  = c0 * E + b
    den2  = 1 - E
    
    return coef * num1 / den1 / den2

