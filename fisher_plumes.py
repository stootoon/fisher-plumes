import os, sys
import numpy as np
from scipy.stats import mannwhitneyu, mode
from scipy.special import betainc
import logging
from copy import deepcopy

import pdb
#sys.path.append(os.environ["CFDGITPY"])
import fisher_plumes_tools as fpt
import utils

import boulder
import crick

logger = utils.create_logger("fisher_plumes")
logger.setLevel(logging.DEBUG)

INFO  = logger.info
DEBUG = logger.debug

class FisherPlumes:

    def __init__(self, sim_name, pitch_in_um = 1, freq_max = np.inf, pairs_mode = "unsigned", n_bootstraps=0, random_seed = 0, **kwargs):
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
            self.pitch_in_um = other.pitch_in_um
            self.wnd   = other.wnd
            self.freq_max = other.freq_max
            for fld in ["fs", "dimensions"]:
                self.__dict__[fld] = other.sim0.__dict__[fld]
            INFO(f"Copied data fields from FisherPlumes object.")
        else:
            INFO(f"****** LOADING {sim_name=} ******")
            if sim_name == "boulder16":
                which_coords, kwargs = utils.get_args(["which_coords"], kwargs)            
                self.sims, self.pairs = boulder.load_sims(which_coords, pairs_mode = pairs_mode, **kwargs)
            elif sim_name == "n12dishT": 
                self.sims, self.pairs = crick.load_sims(sim_name, pairs_mode = pairs_mode,  **kwargs)               
            else:
                raise ValueError(f"Don't know how to load {sim_name=}.")
            self.name         = sim_name
            self.n_bootstraps = n_bootstraps
            self.random_seed  = random_seed
            self.yvals = np.array(sorted(list(self.sims.keys())))
            self.pairs_mode   = pairs_mode            
            self.wnd = None
            self.freq_max = freq_max
            self.pitch_in_um = pitch_in_um            
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

        wnd    = self.wnd
        n_probes = utils.d1(self.sims).data.shape[1]
        INFO(f"Computing coefficients for {n_probes} probes.")        
        self.ss, self.cc, self.tt = [{} for _ in range(n_probes)],[{} for _ in range(n_probes)],[{} for _ in range(n_probes)]
        
        detrender = lambda x: fpt.Detrenders.tukey_normalizer(x, tukey_param)
        for src in self.sims:
            for i in range(n_probes):
                ss, cc, tt = fpt.compute_sin_cos_stft(self.sims[src].data[:,i], istart, wnd, wnd//2, detrender=detrender, **kwargs);
                self.ss[i][src], self.cc[i][src], self.tt[i][src] = [self.bootstrap(fld, dim=0) for fld in [ss,cc,tt]] # The same random seed is used every time so the ss, cc and tt line up after bootstrapping

    def compute_vars_for_freqs(self):
        INFO("Computing variances for harmonics.")
        sc_all = [np.concatenate([xkk for Fld in [ss, cc] for src, xkk in Fld.items()], axis = 1) # axis = 1: concatenate long the time axis.
                  for ss,cc in zip(self.ss, self.cc)] # Loop over probes
        # sc_all[i] is now a (# bootstraps, # sine coefs + # cos coefs, # freqs) matrix.
        self.vars_for_freqs = [np.var(sc_alli,axis=1) for sc_alli in sc_all]

    def compute_correlations_from_trig_coefs(self):
        INFO("Computing correlations from trig coefficients.")
        self.rho    = [{d:np.concatenate([(ss[d1]*ss[d2] + cc[d1]*cc[d2])/2 for (d1,d2) in pairsd], axis=1).transpose([0,2,1]) # bs x freqs x windows. 
                       for d,pairsd in self.pairs.items()} for ss,cc in zip(self.ss, self.cc)]

    def create_pooling_functions(self):
        INFO("Creating pooling functions.")
        # Takes a window size and a frequency INDEX and pools the data for the sine and cosine coefficients from each source
        #Create one function for each probe
        self.pool_cs_data = [(lambda fi:{src:np.hstack([ss[src][:,:,fi], cc[src][:,:,fi]]) for src in ss}) for ss,cc in zip(self.ss, self.cc)]
        # Given pooled cs data, combines the data across elements of all distance pairs for a given distance and returns the resultsing two vectors.
        self.pool_cs_data_for_distance = lambda cs_data, d: np.stack([np.hstack([cs_data[y0] for y0,y1 in self.pairs[d]]),
                                                                      np.hstack([cs_data[y1] for y0,y1 in self.pairs[d]])],axis=0).transpose([1,0,2]) # Bootstraps x (y0, y1) x windows

    def pool_trig_coefs_for_distance(self, d):
        cs_0 = [np.concatenate([coef_prb[y0] for coef_prb in [ss_prb, cc_prb] for (y0,y1) in self.pairs[d]], axis=1) for (ss_prb,cc_prb) in zip(self.ss, self.cc)]
        cs_1 = [np.concatenate([coef_prb[y1] for coef_prb in [ss_prb, cc_prb] for (y0,y1) in self.pairs[d]], axis=1) for (ss_prb,cc_prb) in zip(self.ss, self.cc)]
        cs_pooled = [np.transpose(np.stack((cs0_prb, cs1_prb), 3), [0,3,1,2]) for (cs0_prb, cs1_prb) in zip(cs_0, cs_1)]
        return cs_pooled
        # cs_pooled[iprb][bs, 2, #windows, fi]
        
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
        compute_variances = lambda X: np.var(np.dot(EE.T,X),axis=1)

        dists = sorted(list(self.pairs.keys()))
        n_probes = len(self.ss)
        self.la, self.mu = [{d:[] for d in dists} for _ in range(n_probes)], [{d:[] for d in dists} for _ in range(n_probes)]
        freqs = np.arange(self.wnd)/self.wnd * self.fs
        if fmax is None: fmax = self.fs/2
        DEBUG(f"{sum(freqs<=fmax)=}.")
        for iprb in range(n_probes):
            for fi,f in enumerate(freqs[freqs<=fmax]):
                pooled_data = self.pool_cs_data[iprb](fi)
                for d in dists:
                    data = self.pool_cs_data_for_distance(pooled_data, d) # bs x (y0, y1) x data
                    vars = np.array([compute_variances(datai) for datai in data]).T # .T so μ is the first row, and λ is the second
                    fi==0 and iprb ==0 and d == dists[0] and (DEBUG(f"{data.shape=}"),DEBUG(f"{vars.shape=}"))
                    # Append the data for this frequency
                    self.la[iprb][d].append(vars[0]) # Projection along (1,1)             
                    self.mu[iprb][d].append(vars[1]) # Projection along (1,-1)
        
        for iprb in range(n_probes):
            for d in dists:
                self.la[iprb][d] = np.array(self.la[iprb][d]).T # .T for bootstraps x data
                self.mu[iprb][d] = np.array(self.mu[iprb][d]).T

        DEBUG(f"{utils.d1(self.la[0]).shape=}")
    
    def compute_pvalues(self, skip_bootstrap = True):
        INFO("Computing p-values.")
        if skip_bootstrap:
            INFO("(Skipping p-value computation for bootstraps.)")
        self.pvals = [
            {d:np.array([
                [np.nan if ((ibs>0) and skip_bootstrap) else fpt.compute_ks_pvalue(lad_bs_f, mud_bs_f, rhod_bs_f) for (lad_bs_f, mud_bs_f, rhod_bs_f) in zip(lad_bs, mud_bs, rhod_bs)]
                for ibs, (lad_bs, mud_bs, rhod_bs) in enumerate(zip(la[d], mu[d], rho[d]))]) for d in rho} for la,mu,rho in zip(self.la, self.mu, self.rho)]

    def compute_r2values(self, skip_bootstrap = True):
        INFO("Computing R^2-values.")
        if skip_bootstrap:
            INFO("(Skipping R^2-value computation for bootstraps.)")
        self.r2vals = [
            {d:np.array([
            [np.nan if ((ibs>0) and skip_bootstrap) else fpt.compute_r2_value(lad_bs_f, mud_bs_f, rhod_bs_f) for (lad_bs_f, mud_bs_f, rhod_bs_f) in zip(lad_bs, mud_bs, rhod_bs)]
                for ibs, (lad_bs, mud_bs, rhod_bs) in enumerate(zip(la[d], mu[d], rho[d]))]) for d in rho} for la,mu,rho in zip(self.la, self.mu, self.rho)]
        
    def compute_la_gen_fit_to_distance(self, dmax=100000):
        INFO(f"Computing generalized exponential fit to distance.")
        dists = np.array(sorted(list(self.la[0].keys())))
        dd    = dists[np.abs(dists)<=dmax]
    
        INFO(f"Using {len(dd)} distances <= {dmax}")
        la_sub = [np.stack([la[d] for d in dists if abs(d) <= dmax],axis=-1) for la in self.la]
        n_bs, n_freqs, n_dists = la_sub[0].shape
        
        INFO(f"Computed λ for {n_freqs} frequencies and {n_dists} distances and {n_bs} bootstraps.")
        if ('vars_for_freqs' not in self.__dict__) or self.vars_for_freqs is None:
            # Loop over the rows of la_sub
            # Each row (la_subi) has the data for one frequency
            self.fit_params = [np.array([[fpt.fit_gen_exp(dd/np.std(dd),la_sub_bsi) for la_sub_bsi in la_sub_bs] for la_sub_bs in la_subi])
                               for la_subi in la_sub]
        else:
            INFO(f"Not fitting amplitudes, instead using given values.")

            self.fit_params = [
                np.array([[
                fpt.DUMP_IF_FAIL(fpt.fit_gen_exp_no_amp, dd/np.std(dd),la_sub_bsi, ampi, extra={"ifreq":ifreq, "ibs":ibs, "iprobe":iprobe})
                for ifreq,  (la_sub_bsi, ampi)             in enumerate(zip(la_sub_bs, 2*vars_for_freqs_bs))]
                for ibs,    (la_sub_bs, vars_for_freqs_bs) in enumerate(zip(la_subi,     vars_for_freqsi))])
                for iprobe, (la_subi, vars_for_freqsi)     in enumerate(zip(la_sub, self.vars_for_freqs))]
            DEBUG(f"{self.fit_params[0].shape=}.")
            DEBUG(f"{self.vars_for_freqs[0].shape=}.")
            # Stack the given amplitudes on top of the learned parameters so that the
            # shapes are the same whether we learn the amplitudes or not.
            self.fit_params = [
                np.stack([2*vars_for_freqsi] + [fit_paramsi[:,:,i] for i in range(fit_paramsi.shape[-1])], axis=2)
                for vars_for_freqsi, fit_paramsi in zip(self.vars_for_freqs, self.fit_params)]

        for iprb in range(len(self.fit_params)):
            self.fit_params[iprb][:,:,1] *= np.std(dd)  # Scale the length scale back to their raw values.
        self.dd_fit = dd

    def compute_fisher_information_at_distances(self, which_ds):
        return [np.array([fpt.compute_fisher_information_for_gen_exp_decay(d,
                                                                          fit_params[:,:,1],
                                                                          fit_params[:,:,2],
                                                                          fit_params[:,:,3],
                                                                          vars_for_freqs                                                                       
        ) for d in which_ds]).transpose([1,2,0]) # bs * freq * dists
                for fit_params, vars_for_freqs in zip(self.fit_params, self.vars_for_freqs)]

    def compute_fisher_information(self, d_min = 100, d_max = -1, d_add = [100,200,500,1000,2000,5000]):
        INFO(f"Computing Fisher information (v2).")
        d_vals = [d for d in list(sorted(self.la[0].keys())) if d>0]
        if len(d_add): d_vals += d_add
        d_vals = sorted(list(set(d_vals)))
        if d_min < d_vals[0]: d_vals = [d_min] + d_vals
        if d_max > d_vals[-1]: d_vals.append(d_max)
        INFO(f"Evaluating at distances: {d_vals}.")
        
        self.I_dists = np.array(d_vals)
        self.I = self.compute_fisher_information_at_distances(d_vals)

        n_freqs = utils.d1(self.la[0]).shape[1]
        expected_shape = (self.n_bootstraps+1, n_freqs, len(d_vals))
        assert self.I[0].shape == expected_shape, f"{self.I[0].shape=} <> {expected_shape=}."
        DEBUG(f"{self.I[0].shape=} has the expected value.")

        pcs = [5, 50, 95]
        # Compute the percentiles over bootstraps ([1:] in the first dimension)
        self.I_pcs = [{pc:Ipc for (pc, Ipc) in zip(pcs, np.percentile(I[1:], pcs, axis=0))} for I in self.I]
        
        ifreq_max = self.freqs2inds([self.freq_max])[0]

        Isort      = [np.argsort(I[1:][:, 1:ifreq_max+1,:],axis=1) for I in self.I] # Sort frequencies by information
        n_freqs    = Isort[0].shape[1]
        best_ifreqs = [Isorti[:,-1,:] for Isorti in Isort] # Find the most informative frequency for each bootstrap and distance
        res = [mode(best_ifreqsi, keepdims=False) for best_ifreqsi in best_ifreqs] # Find the frequency that was most frequently most informative
        self.I_best_ifreqs = [r.mode + 1 for r in res]
        # The p-values are those of binomial random variable
        # Have a probability of 1/# frequencies
        # and N = #bootstraps trials.
        # We want to see how many times a given frequency would come out on top
        # if it was happening by chance, and that's determined by the Binomial cdf
        # Which is the Incomplete beta function below
        self.I_pvals = [np.array([betainc(ci, self.n_bootstraps - ci + 1, 1./n_freqs) for ci in r.count]) for r in res]
        
    def compute_all_for_window(self, wnd, istart=0, window='boxcar', tukey_param=0.1, dmax=25000, fit_vars = True):
        self.set_window(wnd)
        self.compute_trig_coefs(istart=istart, window=window, tukey_param=tukey_param)
        not fit_vars and self.compute_vars_for_freqs() 
        self.compute_correlations_from_trig_coefs()
        self.compute_lambdas()
        self.compute_pvalues()
        self.compute_r2values()
        self.compute_la_gen_fit_to_distance(dmax=dmax)
        self.compute_fisher_information()
        INFO(f"Done computing all for {wnd=}.")

    def freqs2inds(self, which_freqs):
        # Figures out the indices of the fft that correspond to each of the frequencies we want
        return [int(round(f * self.wnd / self.fs)) for f in which_freqs]

    def inds2freqs(self, which_inds):
        # Figures out the frequencies that indices correspond to
        return self.freqs[which_inds]
    
    def save_snapshots(self, t, data_dir = "."):
        {k:s.save_snapshot(t, data_dir = data_dir) for k, s in self.sims.items()}

    def load_saved_snapshots(self, t, data_dir = "."):
        return {k:s.load_saved_snapshot(t, data_dir = data_dir) for k, s in self.sims.items()}
        
