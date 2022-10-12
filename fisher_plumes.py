import os, sys
import numpy as np
import logging
from copy import deepcopy

#sys.path.append(os.environ["CFDGITPY"])
import fisher_plumes_tools as fpt
import utils

logger = utils.create_logger("fisher_plumes")
logger.setLevel(logging.DEBUG)

INFO  = logger.info
DEBUG = logger.debug

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
            self.pairs_mode = other.pairs_mode            
            self.sim0  = deepcopy(other.sim0)
            self.wnd   = other.wnd
            for fld in ["fs", "dimensions"]:
                self.__dict__[fld] = other.sim0.__dict__[fld]
            INFO(f"Copied data fields from FisherPlumes object.")
        else:
            INFO(f"****** LOADING {sim_name=} ******")
            if sim_name == "boulder16":
                which_coords, kwargs = FisherPlumes.get_args(["which_coords"], kwargs)            
                self.sims, self.pairs = fpt.load_boulder_16_source_sims(which_coords, pairs_mode = pairs_mode, **kwargs)
            elif sim_name == "n12dishT": 
                self.sims, self.pairs = fpt.load_crick(sim_name, pairs_mode = pairs_mode,  **kwargs)               
            else:
                raise ValueError(f"Don't know how to load {sim_name=}.")
            self.name         = sim_name
            self.n_bootstraps = n_bootstraps
            self.random_seed  = random_seed
            self.yvals = np.array(sorted(list(self.sims.keys())))
            self.pairs_mode   = pairs_mode            
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
        self.freqs = np.arange(wnd)/wnd*self.fs
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
        INFO("Computing amplitude of generalized gaussian fits to λ(s).")
        sc_all = np.concatenate([xkk for Fld in [self.ss, self.cc] for src, xkk in Fld.items()], axis = 1) # axis = 1: concatenate long the time axis.  
        # sc_all is now a (# bootstraps, # sine coefs + # cos coefs, # freqs) matrix.
        sc_vars_for_freqs = np.var(sc_all,axis=1)
        self.amps_for_freqs = 2 * sc_vars_for_freqs

    def compute_correlations_from_trig_coefs(self):
        INFO("Computing correlations from trig coefficients.")
        self.rho    = {d:np.concatenate([(self.ss[d1]*self.ss[d2] + self.cc[d1]*self.cc[d2])/2 for (d1,d2) in pairsd], axis=1).transpose([0,2,1]) # bs x freqs x windows. 
                       for d,pairsd in self.pairs.items()}

    def create_pooling_functions(self):
        INFO("Creating pooling functions.")
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
        INFO("Computing lambdas.")
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

    def compute_pvalues(self, skip_bootstrap = True):
        INFO("Computing p-values.")
        if skip_bootstrap:
            INFO("(Skipping p-value computation for bootstraps.)")
        self.pvals = {d:np.array([
            [np.nan if ((ibs>0) and skip_bootstrap) else fpt.compute_ks_pvalue(lad_bs_f, mud_bs_f, rhod_bs_f) for (lad_bs_f, mud_bs_f, rhod_bs_f) in zip(lad_bs, mud_bs, rhod_bs)]
            for ibs, (lad_bs, mud_bs, rhod_bs) in enumerate(zip(self.la[d], self.mu[d], self.rho[d]))]) for d in self.rho}

    def compute_la_gen_fit_to_distance(self, dmax=100000):
        INFO(f"Computing generalized exponential fit to distance.")
        dists = np.array(sorted(list(self.la.keys())))
        dd    = dists[np.abs(dists)<=dmax]
    
        INFO(f"Using {len(dd)} distances <= {dmax}")
        la_sub = np.stack([self.la[d] for d in dists if abs(d) <= dmax],axis=-1)
        n_bs, n_freqs, n_dists = la_sub.shape
        
        INFO(f"Computed λ for {n_freqs} frequencies and {n_dists} distances and {n_bs} bootstraps.")
        if ('amps_for_freqs' not in self.__dict__) or self.amps_for_freqs is None:
            # Loop over the rows of la_sub
            # Each row (la_subi) has the data for one frequency
            self.fit_params = np.array([[fpt.fit_gen_exp(dd/np.std(dd),la_sub_bsi) for la_sub_bsi in la_sub_bs] for la_sub_bs in la_sub])
        else:
            INFO(f"Not fitting amplitudes, instead using given values.")

            self.fit_params = np.array([[fpt.fit_gen_exp_no_amp(dd/np.std(dd),la_sub_bsi, ampi) for (la_sub_bsi, ampi) in zip(la_sub_bs, amps_for_freqs_bs)]
                                        for (la_sub_bs, amps_for_freqs_bs)  in zip(la_sub, self.amps_for_freqs)])
            DEBUG(f"{self.fit_params.shape=}.")
            # Stack the given amplitudes on top of the learned parameters so that the
            # shapes are the same whether we learn the amplitudes or not.
            self.fit_params = np.stack([self.amps_for_freqs, self.fit_params[:,:,0], self.fit_params[:,:,1]], axis=2)
                                         
        self.fit_params[:,:,1] *= np.std(dd)  # Scale the length scale back to their raw values.
        self.dd_fit = dd

    def compute_fisher_information_at_distances(self, which_ds):
        return np.array([fpt.compute_fisher_information_for_gen_exp_decay(d,
                                                                          self.fit_params[:,:,1],
                                                                          self.fit_params[:,:,2]) for d in which_ds])

    def compute_fisher_information(self):
        d_vals = list(self.la.keys())
        I = self.compute_fisher_information_at_distances(d_vals)
        self.fisher_information = {d:Id for d, Id in zip(d_vals, I)}
                                   
    def compute_all_for_window(self, wnd, istart=0, window='boxcar', tukey_param=0.1, dmax=25000, fit_amps = True):
        self.set_window(wnd)
        self.compute_trig_coefs(istart=istart, window=window, tukey_param=tukey_param)
        not fit_amps and self.compute_amps_for_freqs() 
        self.compute_correlations_from_trig_coefs()
        self.compute_lambdas()
        self.compute_pvalues()
        self.compute_la_gen_fit_to_distance(dmax=dmax)
        self.compute_fisher_information()

    def freqs2inds(self, which_freqs):
        # Figures out the indices of the fft that correspond to each of the frequencies we want
        return [int(round(f * self.wnd / self.fs)) for f in which_freqs]
        
    def save_snapshots(self, t, data_dir = "."):
        [s.save_snapshot(t, data_dir = data_dir) for _, s in self.sims.items()]

    def load_saved_snapshots(self, t, data_dir = "."):
        return np.array([s.load_saved_snapshot(t, data_dir = data_dir) for _, s in self.sims.items()])
        
