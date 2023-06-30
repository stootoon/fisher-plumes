import pdb

import os, sys
import numpy as np
from scipy.stats import mannwhitneyu, mode
from scipy.special import betainc
import logging
from copy import deepcopy

from sklearn.linear_model import LinearRegression

import fisher_plumes_tools as fpt
import utils

from units import UNITS

import boulder
import crick
import surrogate

from matplotlib import pyplot as plt

logger = utils.create_logger("fisher_plumes")
logger.setLevel(logging.DEBUG)

INFO  = logger.info
DEBUG = logger.debug

class FisherPlumes:

    def __init__(self, sim_name, copy_stats = False, freq_max = np.inf * UNITS.hertz, pairs_mode = "unsigned", n_bootstraps=0, random_seed = 0, max_time = np.inf * UNITS.s, **kwargs):
        if hasattr(sim_name,"__class__") and sim_name.__class__.__name__ == "FisherPlumes":
            INFO(f"{sim_name=} is a FisherPlumes object named {sim_name.name}.")
            other = sim_name
            copied = self.init_from_dict(other.__dict__)
            INFO(f"Copied {len(copied)} data fields from FisherPlumes object.")
        elif type(sim_name) is dict:
            INFO(f"Initializing from dictionary.")
            copied = self.init_from_dict(sim_name)
            INFO(f"Copied {len(copied)} data fields from supplied dictionary.")            
        elif type(sim_name) is str:
            INFO(f"****** LOADING {sim_name=} ******")
            self.name         = sim_name
            self.pitch_string = f"{self.name}_pitch"
            if self.pitch_string not in UNITS:
                raise KeyError(f"{self.pitch_string} was not found in the units registry. Please define it in 'units.txt'.")                            
            self.pitch = UNITS[self.pitch_string]            
            INFO(f"1 {self.pitch_string} = {(1 * UNITS(f'{self.pitch_string}')).to(UNITS.m)}")
            INFO(f"1 {self.pitch_string} = {(1 * UNITS(f'{self.pitch_string}')).to(UNITS.cm)}")                                    
            INFO(f"1 {self.pitch_string} = {(1 * UNITS(f'{self.pitch_string}')).to(UNITS.mm)}")
            INFO(f"1 {self.pitch_string} = {(1 * UNITS(f'{self.pitch_string}')).to(UNITS.um)}")
            if sim_name == "boulder16":
                which_coords, kwargs = utils.get_args(["which_coords"], kwargs)            
                self.sims, self.pairs_um = boulder.load_sims(which_coords,
                                                             pairs_mode = pairs_mode,
                                                             units = UNITS.m,
                                                             pitch_units = UNITS(self.pitch_string),
                                                             **kwargs)
            elif sim_name in ["n12dishT", "n12T", "n12Tslow", "n16T", "n16Tslow"]+ [f"crimgrid_w{i}" for i in range(1,5)]:
                self.sims, self.pairs_um = crick.load_sims(sim_name,
                                                           pairs_mode = pairs_mode,
                                                           units = UNITS.m,
                                                           pitch_units = UNITS(self.pitch_string),
                                                           max_time = max_time,
                                                           **kwargs)
            elif sim_name.startswith("surr_"):
                which_coords, kwargs = utils.get_args(["which_coords"], kwargs)            
                surr_type = sim_name[5:]
                self.sims, self.pairs_um = surrogate.load_sims(surr_type,
                                                               which_coords,
                                                               pairs_mode = pairs_mode,
                                                               units = UNITS.m,
                                                               pitch_units = UNITS(self.pitch_string),
                                                               **kwargs)
                INFO(f"{list(self.sims.keys())=}")
            else:
                print(crick.list_datasets())
                raise ValueError(f"Don't know how to load {sim_name=}.")
            self.n_bootstraps = n_bootstraps
            self.random_seed  = random_seed
            self.yvals_um     = np.array(sorted(list(self.sims.keys())))
            self.pairs_mode   = pairs_mode            
            self.wnd = None
            self.freq_max = freq_max
            self.sim0 = self.sims[self.yvals_um[0]]
            for fld in ["fs", "dimensions"]:
                self.__dict__[fld] = self.sim0.__dict__[fld]
        else:
            raise ValueError(f"Don't know what to do for {sim_name=}.")
                
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
                
    def copy_data_fields(self, copy_sims = False):
        """
        Return a dictionary of all the data fields in this object.
        A field is considered a data field if it is not callable.
        """
        data = {}
        for fld,val in self.__dict__.items():
            if not callable(val):
                if not copy_sims and fld in ["sim0", "sims"]: continue
                data[fld] = deepcopy(val)
        return data
    
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

    def compute_trig_coefs(self, istart = 0, window = ('boxcar'), z_score = True, **kwargs):
        INFO(f"Computing trig coefficients for {self.name} with {istart=} and {window=} and {z_score=} and {kwargs=} ")
        if self.wnd is None: raise ValueError("Window size is unset. Use set_window to set it.")

        wnd    = self.wnd
        n_probes = utils.d1(self.sims).data.shape[1]
        INFO(f"Computing coefficients for {n_probes} probes.")        
        self.ss, self.cc, self.tt = [{} for _ in range(n_probes)],[{} for _ in range(n_probes)],[{} for _ in range(n_probes)]
        
        for src in self.sims:
            for i in range(n_probes):
                ss, cc, tt = fpt.compute_sin_cos_stft(self.sims[src].data[:,i], istart, wnd, wnd//2, window=window, z_score = z_score, **kwargs);
                self.ss[i][src], self.cc[i][src], self.tt[i][src] = [self.bootstrap(fld, dim=0) for fld in [ss,cc,tt]] # The same random seed is used every time so the ss, cc and tt line up after bootstrapping

    def compute_vars_for_freqs(self):
        """
        Compute the variances of the pooled sine and cosine coefficients for each frequency.
        """
        INFO("Computing variances for harmonics.")
        sc_all = [np.concatenate([xkk for Fld in [ss, cc] for src, xkk in Fld.items()], axis = 1) # axis = 1: concatenate long the time axis.
                  for ss,cc in zip(self.ss, self.cc)] # Loop over probes
        # sc_all[i] is now a (# bootstraps, # sine coefs + # cos coefs, # freqs) matrix.
        self.vars_for_freqs = [np.var(sc_alli,axis=1) for sc_alli in sc_all]

    def compute_correlations_from_trig_coefs(self):
        INFO("Computing correlations from trig coefficients.")
        self.rho    = [{d:np.concatenate([(ss[d1]*ss[d2] + cc[d1]*cc[d2])/2 for (d1,d2) in pairsd], axis=1).transpose([0,2,1]) # bs x freqs x windows. 
                       for d,pairsd in self.pairs_um.items()} for ss,cc in zip(self.ss, self.cc)]

    def pool_trig_coefs_for_distance(self, d):
        cs_0 = [np.concatenate([coef_prb[y0] for coef_prb in [ss_prb, cc_prb] for (y0,y1) in self.pairs_um[d]], axis=1) for (ss_prb,cc_prb) in zip(self.ss, self.cc)]
        cs_1 = [np.concatenate([coef_prb[y1] for coef_prb in [ss_prb, cc_prb] for (y0,y1) in self.pairs_um[d]], axis=1) for (ss_prb,cc_prb) in zip(self.ss, self.cc)]
        cs_pooled = [np.transpose(np.stack((cs0_prb, cs1_prb), 3), [0,3,1,2]) for (cs0_prb, cs1_prb) in zip(cs_0, cs_1)]
        return cs_pooled
        # cs_pooled[iprb][bs, 2, #windows, fi]

    def test_trig_coef_pooling(self):
        DEBUG("Testing trig coef pooling.")
        ss_orig = deepcopy(self.ss)
        cc_orig = deepcopy(self.cc)
        ss = deepcopy(self.ss)
        cc = deepcopy(self.cc)
        for iprb in range(len(ss)):
            for y in ss[iprb].keys():
                spy = ss[iprb][y]
                ss[iprb][y] = self.ss[iprb][y].astype('complex128')
                cc[iprb][y] = self.cc[iprb][y].astype('complex128')        
                for ibs in range(spy.shape[0]):
                    for itime in range(spy.shape[1]):
                        for ifreq in range(spy.shape[2]):
                            ss[iprb][y][ibs,itime,ifreq] = (y*10000 + ibs*100+ifreq+itime/1000.) + 1j*(iprb+1)
                            cc[iprb][y][ibs,itime,ifreq] = (y*10000 + ibs*100+ifreq+itime/1000.) - 1j*(iprb+1)                    

        iprb  = min(len(self.ss)-1,1) 
        dist  = list(self.pairs_um.keys())[1]
        ifreq = 1
        ibs   = min(1,self.n_bootstraps)
        y01   = 1
        DEBUG(f"Testing with {iprb=} {dist=} {ifreq=} {ibs=} {y01=}.")

        self.ss = ss
        self.cc = cc
        pooled  = self.pool_trig_coefs_for_distance(dist)[iprb][:,:,:,ifreq] # bs, (y0,y1), time, freq

        expected  = [[y0,y1][y01]*10000 + ibs*100 + ifreq + itime/1000. + 1j*(iprb+1) for (y0,y1) in self.pairs_um[dist] for itime in range(self.ss[iprb][y0].shape[1])]
        expected += [[y0,y1][y01]*10000 + ibs*100 + ifreq + itime/1000. - 1j*(iprb+1) for (y0,y1) in self.pairs_um[dist] for itime in range(self.ss[iprb][y0].shape[1])]
        expected  = np.array(expected)

        exp_vals = sorted(np.imag(expected))
        obs_vals = sorted(np.imag(pooled[ibs][y01][:]))            
        if not np.allclose(exp_vals, obs_vals):
            plt.plot(exp_vals, label="expected")
            plt.plot(obs_vals, label="observed")
            plt.legend()            
            plt.savefig("mismatch.pdf")
            plt.close()
            DEBUG("Wrote mismatch.pdf")
            raise ValueError("Imaginary values mismatched.")
        else:
            DEBUG("Imaginary values were OK.")

        exp_vals = sorted(np.real(expected))
        obs_vals = sorted(np.real(pooled[ibs][y01][:]))
        if not np.allclose(exp_vals, obs_vals):
            plt.plot(exp_vals, label="expected")
            plt.plot(obs_vals, label="observed")
            plt.legend()
            plt.savefig("mismatch.pdf")
            plt.close()
            DEBUG("Wrote mismatch.pdf")            
            raise ValueError("Real values mismatched.")
        else:
            DEBUG("Real values were OK.")

        # Tests passed so restore the original values
        self.ss = ss_orig
        self.cc = cc_orig

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

        EE    = np.array([[1,-1],[1,1]])/np.sqrt(2)
        compute_variances = lambda X: np.var(np.dot(EE.T,X),axis=1)

        dists = sorted(list(self.pairs_um.keys()))
        n_probes = len(self.ss)
        self.la, self.mu = [{d:[] for d in dists} for _ in range(n_probes)], [{d:[] for d in dists} for _ in range(n_probes)]
        freqs = np.arange(self.wnd)/self.wnd * self.fs
        if fmax is None: fmax = self.fs/2
        DEBUG(f"{sum(freqs<=fmax)=}.")
        for d in dists:
            pooled_data = self.pool_trig_coefs_for_distance(d)                    
            for iprb in range(n_probes):
                for fi,f in enumerate(freqs[freqs<=fmax]):
                    data = pooled_data[iprb][:,:,:,fi] # bs x (y0, y1) x data                    
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
        
    def compute_la_gen_fit_to_distance(self, dmax_um=100000, fit_k = True):
        """
        Computes a generalized exponential fit to the decay of correlations with distance.
        """
        INFO(f"Computing generalized exponential fit to distance.")
        dists = np.array(sorted(list(self.la[0].keys())))
        dd    = dists[np.abs(dists)<=dmax_um]
    
        INFO(f"Using {len(dd)} distances <= {dmax_um} um ")
        la_sub = [np.stack([la[d] for d in dists if abs(d) <= dmax_um],axis=-1) for la in self.la]
        n_bs, n_freqs, n_dists = la_sub[0].shape
        
        INFO(f"Computed λ for {n_freqs} frequencies and {n_dists} distances and {n_bs} bootstraps.")
        
        bounds_dict = {k:(0, np.inf) for k in "askb"}
        if not fit_k:
            bounds_dict["k"] = (1,1+1e-6)
            INFO(f"Not fitting k by using {bounds_dict['k']=:}")
        
        if ('vars_for_freqs' not in self.__dict__) or self.vars_for_freqs is None:
            # Loop over the rows of la_sub
            # Each row (la_subi) has the data for one frequency
            self.fit_params = [np.array([[fpt.fit_gen_exp(dd/np.std(dd),la_sub_bsi, bounds_dict = bounds_dict) for la_sub_bsi in la_sub_bs] for la_sub_bs in la_subi])
                               for la_subi in la_sub]
        else:
            INFO(f"Not fitting amplitudes, instead using given values.")

            self.fit_params = [
                np.array([[
                fpt.DUMP_IF_FAIL(fpt.fit_gen_exp_no_amp, dd/np.std(dd),la_sub_bsi, ampi, bounds_dict = bounds_dict, extra={"ifreq":ifreq, "ibs":ibs, "iprobe":iprobe})
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

    def compute_fisher_information_estimates_at_distances(self, which_ds):
        return [np.array([fpt.compute_fisher_information_estimates_for_gen_exp_decay(d,
                                                                          fit_params[:,:,1],
                                                                          fit_params[:,:,2],
                                                                          fit_params[:,:,3],
                                                                          vars_for_freqs                                                                       
        ) for d in which_ds]).transpose([1,2,3,0]) # ests * bs * freq * dists
                for fit_params, vars_for_freqs in zip(self.fit_params, self.vars_for_freqs)]
    
    def compute_fisher_information(self,
                                   d_min = 100 , d_max = -1, d_add = [100,200,500,1000,2000,5000],
                                   pcs = [5, 50, 95]
                                   ):
        INFO(f"Computing Fisher information.")
        d_vals = [d for d in list(sorted(self.la[0].keys())) if d>0]
        if len(d_add): d_vals += d_add
        d_vals = sorted(list(set(d_vals)))
        if d_min < d_vals[0]: d_vals = [d_min] + d_vals
        if d_max > d_vals[-1]: d_vals.append(d_max)
        INFO(f"Evaluating at distances: {d_vals}.")
        
        self.I_dists = np.array(d_vals)
        self.I = self.compute_fisher_information_at_distances(d_vals)
        self.Ilow, self.Ihigh = zip(*self.compute_fisher_information_estimates_at_distances(d_vals))

        n_freqs = utils.d1(self.la[0]).shape[1]
        expected_shape = (self.n_bootstraps+1, n_freqs, len(d_vals))
        assert self.I[0].shape == expected_shape, f"{self.I[0].shape=} <> {expected_shape=}."
        DEBUG(f"{self.I[0].shape=} has the expected value.")
        # Compute the percentiles over bootstraps ([1:] in the first dimension)
        self.I_pcs = [{pc:Ipc for (pc, Ipc) in zip(pcs, np.percentile(I[1:], pcs, axis=0))} for I in self.I]

    def regress_information_on_frequency(self, freq_min = 0 * UNITS.Hz, freq_max = None):
        if freq_max is None: freq_max = self.freq_max
        INFO(f"Regressing fisher information on frequency between {freq_min=:} and {freq_max=:}.")
        self.reg_freq_range = (freq_min, freq_max)
        n_bs, n_I_freq, n_d = self.I[0].shape    
        ind   = (self.freqs >= freq_min) & (self.freqs <= freq_max)
        freqs = self.freqs.to(UNITS.Hz).magnitude
        lr  = LinearRegression()
        self.reg_coefs = []    
        for I in self.I:
            reg_coefs = []
            for Ibs in I:
                for i in range(n_d):
                    yy = Ibs[ind[:n_I_freq],i]
                    xx = freqs[ind][:len(yy)]
                    ivld = ~np.isnan(yy) & (yy>0) & ~np.isinf(yy)
                    if len(ivld)>=2:
                        lr.fit(xx[ivld].reshape(-1,1), np.log10(yy[ivld]))
                        reg_coefs.append([lr.intercept_] + list(lr.coef_))
                    else:
                        reg_coefs.append([np.nan]*2)
    
            reg_coefs = np.array(reg_coefs).reshape(n_bs, n_d, 2)
            self.reg_coefs.append(reg_coefs)        

    def regress_length_constants_on_frequency(self, freq_min):
        ind_use = (self.freqs >= freq_min)  & (self.freqs <= self.freq_max)
        INFO(f"Regressing length constants on frequency, using {freq_min=:g}: {self.freqs[ind_use][0]:g} - {self.freqs[ind_use][-1]:g}")        
        ff = self.freqs[ind_use].to(UNITS.Hz).magnitude.reshape(-1,1)
        d_scale = self.pitch.to(UNITS.um).magnitude
        lr = LinearRegression()
        pack_coefs = lambda lr: [lr.intercept_, lr.coef_[0]]
        self.coef_γ_vs_freq = [np.array([pack_coefs(lr.fit(ff,γbs[ind_use[:len(γbs)]])) for γbs in fp[:, :, 1]/d_scale]) for fp in self.fit_params]            
    
    def compute_all_for_window(self, window_length, window_shape=('boxcar'), istart=0, dmax_um=np.inf, fit_vars = False, fit_k = True, z_score = True):
        INFO(f"STARTING COMPUTATION.")
        wnd = int(window_length.to(UNITS.s).magnitude * self.fs.to(UNITS.Hz).magnitude)
        INFO(f"Setting window to {wnd} samples.")
        self.set_window(wnd)
        self.compute_trig_coefs(istart=istart, window=window_shape, z_score = z_score)
        not fit_vars and self.compute_vars_for_freqs() 
        self.compute_correlations_from_trig_coefs()
        self.compute_lambdas()
        self.compute_pvalues()
        self.compute_r2values()
        self.compute_la_gen_fit_to_distance(dmax_um=dmax_um, fit_k = fit_k)
        self.regress_length_constants_on_frequency(freq_min = 2 * UNITS.Hz)
        self.compute_fisher_information()
        self.regress_information_on_frequency(freq_min = 1 * UNITS.Hz)
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
        return {k:s.load_saved_snapshot(t, data_dir = data_dir) for k, s in self.sims.items()} if hasattr(self.sim0, "load_saved_snapshot") else None
        
