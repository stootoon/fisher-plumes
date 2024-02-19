import os, sys
import logging
import numpy as np
import pickle
from builtins import sum as bsum
from scipy.signal import stft, tukey, get_window
from scipy.stats  import kstest
from scipy.optimize import curve_fit, minimize
from matplotlib import pylab as plt


import utils

logger = utils.create_logger("fisher_plumes_tools")
logger.setLevel(logging.DEBUG)

INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

def pool_sorted_keys(d, res=0):
    """
    Given a sorted list of keys, group them into lists of keys that are <= res apart.
    Inputs: 
    d = sorted list of keys
    res = resolution, i.e. the maximum distance between keys in a group.
    Output:
    dd = list of lists of keys, where the keys in each list are <= res apart.S
    """
    if res==0:
        dd = [[di] for di in d]
    else:
        dd = [[d[0]]]
        for i in range(1, len(d)):
            dist = abs(d[i] - dd[-1][-1])
            print(f"{d[i]=}, {dd[-1][-1]=}, {dist=}, {res=}")
            if dist <= res:
                print(f"Adding {d[i]} to {dd[-1]}")
                dd[-1].append(d[i])
            else:
                print(f"Creating new group for {d[i]}")
                dd.append([d[i]])
    return dd

class Coords2D_um:
    def __init__(self, y, x=0, L=10**8):
        self.L = L
        assert -L < y < L, f"y={y} must be in the range (-{L}, {L})."
        assert -L < x < L, f"x={x} must be in the range (-{L}, {L})."

        self.y = int(y)
        self.x = int(x)
    def __repr__(self):
        return f"Coords2D_um({self.y=}, {self.x=})"
    def __str__(self):
        return f"(y={self.y}, x={self.x}) um"
    def __eq__(self, other):
        return self.y == other.y and self.x == other.x
    # Implement subtraction
    def __sub__(self, other):
        return Coords2D_um(self.y - other.y, self.x - other.x, self.L)
    def __add__(self, other):
        return Coords2D_um(self.y + other.y, self.x + other.x, self.L)
    def __lt__(self, other):
        return self.y < other.y or (self.y == other.y and self.x < other.x)
    def __le__(self, other):
        return (self < other) or (self == other)
    def __gt__(self, other):
        return not (self <= other)
    def __ge__(self, other):
        return not (self < other)
    def __hash__(self):
        return hash((self.y, self.x))
    def norm(self):
        return np.sqrt(self.y**2 + self.x**2)    
    def __abs__(self):
        return self.norm()
    def abs_src(self):
        return Coords2D_um(abs(self.y), abs(self.x), self.L)
                
def compute_pairs(yvals, pairs_mode="signed", pair_resolution = 0):
    if type(yvals[0]) is not Coords2D_um:
        INFO(f"Converting {len(yvals)=} yvals to Coords2D_um.")
        # Convert to Coords2D_um
        if hasattr(yvals[0], "__len__"):
            INFO(f"Assuming that yvals are (y,x) pairs.")
            yvals = [Coords2D_um(y, x) for x,y in yvals]
        else:
            INFO(f"Assuming that yvals are y values only.")
            yvals = [Coords2D_um(y) for y in yvals]
            
    INFO(f"Computing pairs for {len(yvals)=} from {min(yvals)} to {max(yvals)} using {pairs_mode=}.")
    nyvals = len(yvals)
    pairs = {}
    if pairs_mode == "signed":
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
                k = (y2-y1).abs_src() # Will be a Coords2D_um object
                if k not in pairs:
                    pairs[k] = [(y1,y2), (y2,y1)]
                else:
                    pairs[k].append((y1,y2))
                    pairs[k].append((y2,y1))                    
    elif pairs_mode == "sym":
        for i in range(nyvals//2+ 1):
            i1, i2 = i, nyvals - 1 - i
            y1, y2 = yvals[i1], yvals[i2]
            pairs[y1-y2]= [(y1,y2)]
            pairs[y2-y1]= [(y2,y1)]
        z = yvals[0]
        pairs[z - z] = [(y,y) for y in yvals]
    else:
        raise ValueError(f"Don't know what to do for {pairs_mode=}.")
    INFO(f"Pooling data across pair distances that are <= {pair_resolution} apart.")
    d = sorted(list(pairs.keys()))
    dd = pool_sorted_keys(d, pair_resolution) # Returns grouped distances
    DEBUG(f"{dd=}")
    pr = pair_resolution if pair_resolution > 0 else 1
    # kd = [int(np.round(np.mean(ddi)/pr)*pr) for ddi in dd]
    kd = [ddi[0] for ddi in dd]
    pairs1 = {}
    for grp,kdi in zip(dd,kd):
        DEBUG(f"key={kdi}: grouped distances = {grp}")
        pairs1[kdi] = bsum([pairs[grpj] for grpj in grp], [])

    INFO(f"{len(pairs)} pair distances before pool, {len(pairs1)} pair distances after pooling.")
    pairs = pairs1
    INFO("Removing duplicates in pairs dictionary.")
    pairs = {d:list(set(p)) for d,p in pairs.items()} # Remove any duplicates
    return pairs, d

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
    def windowed_then_zscored(x, w):
        y = Detrenders.windowed(x,w)
        return (y - np.mean(y,axis=-1)[:,np.newaxis])/(np.std(y,axis=-1)[:,np.newaxis]+1e-12)
    
    @staticmethod
    def windowed(x, w):
        y = x*w[np.newaxis,:]        
        return y

    @staticmethod
    def mean_subtract(x):
        y = x.copy()
        return (y - np.mean(y,axis=-1)[:,np.newaxis])
    
    @staticmethod
    def identity(x):
        return x

def compute_sin_cos_stft(data, istart, wnd, ov, x_only = False, force_nonnegative=False, window = ('boxcar'), z_score = True):
    block_starts = list(range(istart, len(data)-wnd, wnd-ov))
    
    x = np.copy(data[block_starts[0]:block_starts[-1]+wnd])
    if force_nonnegative: x[x<0] = 0        
    if x_only: return x

    w = get_window(window, wnd)
    detrender = lambda x: (Detrenders.windowed_then_zscored if z_score else Detrenders.windowed)(x, w)    
    freqs,times, S = stft(x, fs = 1, window='boxcar', # This has to be boxcar, the windowing happens in the detrender
                          nperseg=wnd, noverlap=ov, detrend= detrender,
                          boundary=None, padded=False)
    n =  np.arange(wnd//2+1)
    n = 1 + (n > 0)*(n != wnd//2)
    c =  np.real(S)*n[:, np.newaxis]
    s = -np.imag(S)*2
    return s.T, c.T, times.astype(int)

# CDF of the asymmetric laplacian
alaplace_cdf = lambda la, mu, x: (mu/(la + mu + 1e-12))*np.exp(-2*np.abs(x)/(mu+1e-12))*(x<=0)+ ((mu + la*(1 - np.exp(-2*np.abs(x)/(la+1e-12))))/(mu+la+1e-12))*(x>0)

# COMPUTE P VALUE FOR THE KOLMOGOROV SMIRNOV TEST
compute_ks_pvalue = lambda la, mu, x: kstest(x, lambda x: alaplace_cdf(la, mu, x)).pvalue

# COMPUTE R2 VALUE COMPARING THE PREDICTED AND FIT CDFs
r2fun = lambda x, y: 1 - np.nanmean((x - y)**2)/np.var(x)
tvfun = lambda x, y: 1 - np.nanmax(np.abs(x - y))

def compare_cdfs(cdf_fun, x, cmp_fun = r2fun, n=None):
    if n is None:
        # Use the data distribution
        x_vals   = np.sort(x)        
        cdf_vals = np.arange(1,len(x)+1)/len(x)
    else:
        # Split the range of the data into n evenly spaced points
        x_vals   = np.linspace(np.min(x), np.max(x), n)
        cdf_vals = np.array([np.mean(x <= xj) for xj in x_vals])
    return cmp_fun(cdf_vals, cdf_fun(x_vals))
    
def compute_r2_value(cdf_fun, x, n=None):
    return compare_cdfs(cdf_fun, x, cmp_fun = r2fun, n=n)

def compute_tv_value(cdf_fun, x, n=None):
    return compare_cdfs(cdf_fun, x, cmp_fun = tvfun, n=n)

gen_exp     = lambda d, a, s, k, b: (a - b) * np.exp(-np.abs(d/s)**k) + b
fit_gen_exp = lambda d, la, bounds_dict = None: curve_fit(gen_exp, d, la,
                                      p0=[np.max(la),1,1,0],
                                      maxfev=5000,
                                                          bounds=(0, np.inf) if bounds_dict is None else list(zip(*[bounds_dict[k] for k in "askb"])))[0]
fit_gen_exp_no_amp = lambda d, la, a, bounds_dict = None: curve_fit(lambda d, s, k, b: gen_exp(d, a, s, k, b),
                                                                    d, la,
                                                                    p0=[1,1,0],
                                                                    maxfev=5000,                                                
                                                                    bounds=(0, np.inf) if bounds_dict is None else list(zip(*[bounds_dict[k] for k in "skb"])))[0]
def DUMP_IF_FAIL(f, *args, extra= {}, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        DEBUG(f"DUMP_IF_FAIL caught an exception.")
        with open("dumpiffail.p","wb") as f:
            pickle.dump({"args":args, "kwargs":kwargs, "extra":extra}, f)
        DEBUG(f"Wrote args and kwargs to dumpiffail.p.")
        raise e
              
    

def compute_fisher_information_for_gen_exp_decay(s, γ, k, b, σ2):
    sn    = s/γ
    c0    = (2*σ2 - b)
    E     = np.exp(-sn**k)

    coef  = 2 * k**2 / γ**2 
    num1  = c0 * (sn**(k-1) * E)**2
    den1  = c0 * E + b
    den2  = 1 - E
    
    return coef * num1 / den1 / den2

def compute_fisher_information_estimates_for_gen_exp_decay(s, γ, k, b, σ2):
    coef = k**2 / γ ** k
    coef *= (2 * σ2 - b)/σ2
    coef = np.clip(coef, 1e-16, np.inf)
    logIlow = np.log(coef) + (k-2)*np.log(s)
    Ilow    = np.exp(logIlow)

    coef     = 2 * k**2 / γ ** (2*k) * (2*σ2 - b)/b
    coef = np.clip(coef, 1e-16, np.inf)    
    logIhigh = np.log(coef) + (2*k-2) * np.log(s) - 2/γ**k * np.exp(k * np.log(s))
    Ihigh    = np.exp(logIhigh)
    
    return Ilow, Ihigh


def get_window_name(wnd_sh): return wnd_sh[0] if type(wnd_sh) is tuple else wnd_sh
    
