import os, sys
import numpy as np
import logging
from copy import deepcopy

#sys.path.append(os.environ["CFDGITPY"])
import fisher_plumes_tools as fpt
import utils

logger = utils.create_logger("fisher_plumes")
logger.setLevel(logging.DEBUG)

INFO = logger.info

class FisherPlumes:

    @staticmethod
    def get_args(req_args, kwargs):
        vals = []
        for req in req_args:
            if req not in kwargs:
                raise ValueError(f"{req} not found in kwargs.")
            vals.append(kwargs[req])
            del kwargs[req]
        return *vals, kwargs

    def __init__(self, sim_name, pairs_mode = "unsigned", n_bootstraps=0, random_seed = 0, **kwargs):
        if type(sim_name) is not str:
            INFO(f"{sim_name=} was not a string, assuming it's a FisherPlumes object.")
            other = sim_name
            INFO(f"Attempting to copy data fields.")
            self.name         = other.name
            self.n_bootstraps = other.n_bootstraps
            self.random_seed  = other.random_seed
            self.sims  = deepcopy(other.sims)
            self.pairs = deepcopy(other.pairs)
            self.yvals = deepcopy(other.yvals)
            self.sim0  = deepcopy(other.sim0)
            self.wnd   = other.wnd
            for fld in ["fs", "dimensions"]:
                self.__dict__[fld] = other.sim0.__dict__[fld]
            INFO(f"Copied data fields from FisherPlumes object.")
        else:
            INFO(f"****** LOADING {sim_name=} ******")
            if sim_name == "boulder16":
                which_coords, kwargs = FisherPlumes.get_args(["which_coords"], kwargs)            
                self.sims, self.pairs = fpt.load_boulder_16_source_sims(which_coords, **kwargs)            
            else:
                raise ValueError(f"Don't know how to load {sim_name=}.")

            self.name         = sim_name
            self.n_bootstraps = n_bootstraps
            self.random_seed  = random_seed
            self.yvals = np.array(sorted(list(self.sims.keys())))
            self.wnd = None
            self.sim0 = self.sims[self.yvals[0]]
            for fld in ["fs", "dimensions"]:
                self.__dict__[fld] = self.sim0.__dict__[fld]

    def bootstrap(self, X, dim, to_array = True):
        np.random.seed(self.random_seed)

        n_data = X.shape[dim]

        perm  = list(range(len(X.shape)))
        perm[0], perm[dim] = dim, 0
        X_perm = X.transpose(perm)
        X_bs   = [X]        
        for i in range(self.n_bootstraps):
            Xi = deepcopy(X_perm)[np.random.choice(n_data, n_data)]
            X_bs.append(Xi.transpose(perm))

        return np.array(X_bs) if to_array else X_bs
    

    def set_window(self, wnd):
        self.wnd = wnd        
        INFO(f"Window set to {self.wnd=}.")

    def compute_trig_coefs(self, istart = 0, tukey_param = 0.1, **kwargs):
        INFO(f"Computing trig coefficients for {self.name} with {istart=} and {tukey_param=} and {kwargs=}")
        if self.wnd is None:
            raise ValueError("Window size is unset. Use set_window to set it.")

        wnd = self.wnd
        self.ss, self.cc, self.tt = {}, {}, {}
        detrender = lambda x: fpt.Detrenders.tukey_normalizer(x, tukey_param)
        for src in self.sims:
            n_data = self.sims[src].data.shape[1]
            for i in range(n_data):
                key = (src) if n_data == 1 else (src, i)
                ss, cc, tt = fpt.compute_sin_cos_stft(self.sims[src].data[:,i], istart, wnd, wnd//2, detrender=detrender, **kwargs);
                self.ss[key], self.cc[key], self.tt[key] = [self.bootstrap(fld, dim=0) for fld in [ss,cc,tt]]

    def compute_amps_for_freqs(self):
        sc_all = np.concatenate([xkk for Fld in [self.ss, self.cc] for src, xkk in Fld.items()], axis = 1) # axis = 1: concatenate long the time axis.  
        # sc_all is now a (# bootstraps, # sine coefs + # cos coefs, # freqs) matrix.
        sc_vars_for_freqs = np.var(sc_all,axis=1)
        self.amps_for_freqs = 2 * sc_vars_for_freqs

    def compute_correlations_from_trig_coefs(self):
        self.rho    = {d:np.concatenate([(self.ss[d1]*self.ss[d2] + self.cc[d1]*self.cc[d2])/2 for (d1,d2) in pairsd], axis=1)
                       for d,pairsd in self.pairs.items()}

    def create_pooling_functions(self):
        # Takes a window size and a frequency INDEX and pools the data for the sine and cosine coefficients from each source
        #pool_cs_data = lambda fi:{y:np.array([cc[src][:,fi], ss[src][:,fi]]).flatten() for src in self.ss}
        self.pool_cs_data = lambda fi:{src:np.hstack([self.ss[src][:,:,fi], self.cc[src][:,:,fi]]) for src in self.ss}
        # Given pooled cs data, combines the data across elements of all distance pairs for a given distance and returns the resultsing two vectors.
        self.pool_cs_data_for_distance = lambda cs_data, d: np.stack([np.hstack([cs_data[y0] for y0,y1 in self.pairs[d]]),
                                                                      np.hstack([cs_data[y1] for y0,y1 in self.pairs[d]])],axis=0).transpose([1,0,2]) # Bootstraps x (y0, y1) x windows
            
            
    def compute_lambdas(self, fmax = None):
        """ 
        Computes bivariate normal fits to the pooled sine and cosine data.
        Returns results in mu, la, dists
        Each are dictionaries indexed as (WND, FREQ, DIST).
        WND is the window size in samples.
        FREQ is the integer frequency in Hz.
        mu[dist][wnd][ifreq] is an array containing the value fo lambda0
        for for the specified window, indexed frequency, and distance.
        """
        self.create_pooling_functions()

        EE    = np.array([[1,-1],[1,1]])/np.sqrt(2)
        compute_variances = lambda X: np.sort(np.var(np.dot(EE.T,X),axis=1))

        dists = sorted(list(self.pairs.keys()))
        self.la, self.mu = {d:[] for d in dists}, {d:[] for d in dists}
        freqs = np.arange(self.wnd)/self.wnd * self.fs
        if fmax is None: fmax = self.fs/2
        for fi,f in enumerate(freqs[freqs<=fmax]):
            pooled_data = self.pool_cs_data(fi)
            for d in dists:
                data = self.pool_cs_data_for_distance(pooled_data, d) # bs x (y0, y1) x data
                vars = np.array([compute_variances(datai) for datai in data]).T # .T so μ is the first row, and λ is the second
                # Append the data for this frequency
                self.mu[d].append(vars[0]) #[0], [1] because the lambda values are sorted                
                self.la[d].append(vars[1])
    
        for d in dists:
            self.la[d] = np.array(self.la[d]).T # .T for bootstraps x data
            self.mu[d] = np.array(self.mu[d]).T         

        
