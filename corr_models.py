# This file contains the various models that can be fit to the data.
import numpy as np
from matplotlib import pylab as plt
from collections import namedtuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import logging
import yaml
import pickle
import pandas as pd

from scipy.stats import    norm as norm_dist
from scipy.stats import   gamma as gamma_dist
from scipy.stats import   expon as expon_dist
from scipy.stats import   geninvgauss
from scipy.special import gamma as gamma_fun
from scipy.special import digamma, kv
from scipy.optimize import minimize

import fisher_plumes_tools as fpt
import utils

logger = utils.create_logger("corr_models")
logger.setLevel(logging.DEBUG)
INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

rand = np.random.rand
randn= np.random.randn

params2str  = lambda params, flds: ", ".join([f"{p}={v:<.2g}" for p,v in params._asdict().items() if p in flds])

params_all_close = lambda p1, p2: np.allclose([p1.__getattribute__(p) for p in p1._fields], 
                                              [p2.__getattribute__(p) for p in p1._fields],
                                              rtol=0, atol=1e-6)
hasfield = lambda p, fld: fld in p._fields

def kvv(v, z, ε = 1e-6):
    """
    Numerically estimate the derivative of the modified Bessel function of the second kind
    with respect to the order v.
    """
    return (kv(v+ε, z) - kv(v-ε, z))/(2*ε)

def cdf_data(y, n = 1001):
    ys  = np.array(sorted(y))
    xv  = np.linspace(ys[0], ys[-1], n)
    cdf = np.array([np.mean(ys<=x) for x in xv])
    return cdf, xv
        
def plot_data(y, labs, params = None, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.plot(y,color="gray")
    plt.plot(np.where(labs>0)[0],  y[labs>0],  'b.', label="pos")
    plt.plot(np.where(labs<0)[0],  y[labs<0],  'r.', label="neg")
    plt.plot(np.where(labs==0)[0], y[labs==0], 'y.', label="noise")
    plt.legend()
    if params:
        plt.title(params2str(params))

def plot_cdfs(y, mdls = [], labs = [], figsize=None, n = 1001, gof_fun = fpt.compute_r2_value, diffs = False, legend_args = {}):
    cdf_true, xv = cdf_data(y, n = n)
    if not diffs:
        plt.plot(xv, cdf_true, "-",label="data" )

    if len(labs) == 0:
        labs = [str(mdl).replace("(", "\n(") for mdl in mdls]
        
    for mdl,lab in zip(mdls, labs):
        INFO(f"Plotting cdf for model: " + lab.replace("\n",""))        
        cdf_mdl = mdl.cdf(xv)
        gof     = gof_fun(mdl.predict, y, n = n)
        label   = f"{lab} - data" if diffs else lab
        plt.plot(xv, cdf_mdl if not diffs else cdf_mdl - cdf_true, "-", label=f"{label}: {gof:.3f}")

    plt.xlabel("x")
    plt.ylabel("P(data<x)")
    plt.legend(**legend_args)

def fixed_point_iterate(f, x0, max_iter = 1000, tol = 1e-6, damping = 0.5, verbose = False):
    x = x0
    for i in range(max_iter):
        x_new = f(x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = (1 - damping)*x_new + damping*x
        if any(np.isnan(x)):
            break

    status = 0
    if any(np.isnan(x)):
        raise Exception("Nans encountered.")
        status = 2
    elif i == max_iter - 1:
        WARN("Not converged.")
        status = 1
    else:
        INFO(f"Converged to fixed point {x} in {i:>4d} iterations.")
        
    return x_new, status

class Exponential: #(BaseEstimator):
    Params = namedtuple('Params', ['λ', 'μ'])
    Params.__qualname__ = "Exponential.Params" # This is needed for pickle to work.
    
    def cdf(self, x, params = None):
        if params is None: params = self.params
        return fpt.alaplace_cdf(2*params.λ, 2*params.μ, x)

    @staticmethod
    def gen_data(M, params):
        λ, μ = params.λ, params.μ
        y = np.zeros(M)
        pp= λ/(λ + μ)
        ip= rand(M)<pp
        # Set y at ip to be sampled from an exponential with mean λ
        y[ip] = np.random.exponential(λ, sum(ip))
        # Set y at ~ip to be sampled from an exponential with mean μ
        y[~ip]= -np.random.exponential(μ, sum(~ip))       
        labs  = np.sign(y)
        return y, labs

    def __init__(self, init_params = None, min_μ = 1e-6, name = "Model"):
        self.name  = name
        self.min_μ = min_μ
        self.init_params = init_params
        self.params      = init_params

    def __repr__(self):
        return f"{self.name}, {self.__class__.__name__}(Params({params2str(self.params, self.Params._fields)}))"

    def __str__(self):
        return f"{self.name}, {self.__class__.__name__}({params2str(self.params, self.Params._fields)})"

    def init_from_fit(self, y):
        INFO("Initializing from fit.")        
        y = y.flatten()

        INFO(f"Found {sum(y>=0):d} positive, {sum(y<0):d} negative values.")
        λ = np.mean(y[y>=0])
        INFO(f"Fit exponential distribution to positive values, λ={λ:.3g}.")
        μ = np.mean(abs(y[y<0]))
        INFO(f"Fit exponential distribution to negative values, μ={μ:.3g}.")

        self.init_params = self.Params(λ=λ, μ=μ)
        self.params      = self.init_params
        INFO(f"Parameters initialized to {params2str(self.params, self.Params._fields)}.")
        
        
    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6):
        INFO(f"Fitting exponential model {self.name}.")
        assert y is None, "y must be None"

        y = X.flatten()

        # Initialize
        M     = len(y)
        if self.init_params is None:
            self.init_from_fit(y)
        λ_old, μ_old = self.init_params.λ, self.init_params.μ
    
        for i in range(max_iter):
            ip  = (y>=0)
            i_n = (y<0)    
            n_p = sum(ip)
            nn  = sum(i_n)
            n   = n_p + nn
            ypm = np.mean(y[ip]) if n_p else 0
            ynm = np.mean(abs(y[i_n])) if nn else 0
            λ   = 1/(n+1e-8) * (n_p*ypm + np.sqrt(n_p * nn * ypm * ynm))
            μ   = max(1/(n + 1e-8) * (nn*ynm + np.sqrt(n_p * nn * ypm * ynm)), self.min_μ)        

            self.params = self.Params(λ=λ, μ=μ)
            # Print the values at the current iteration, including the iteration number
            DEBUG(f"Iter {i:>4d}: n+={sum(ip):>4d}, n-={sum(i_n):>4d} " + params2str(self.params, self.Params._fields))
            # Check convergence
            if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol:
                INFO(f"Converged in {i:>4d} iterations to n+={sum(ip):>4d}, n-={sum(i_n):>4d} " + params2str(self.params, self.Params._fields))
                break
            λ_old = λ
            μ_old = μ

        self.labs = np.sign(y)        
        return self

    def predict(self, X):
        # Return the cdf at the values in X.
        return self.cdf(X.flatten(), self.params)

    def score(self, X, y = None, cmp_fun = fpt.r2fun, n = 1001):
        # Compute the similarity between the cdf of the data and the cdf of the model.
        assert y is None, "y must be None"
        y = X.flatten() # The data 
        return fpt.compare_cdfs(self.predict, y, cmp_fun = cmp_fun, n = n)
    
class IntermittentExponential(Exponential):
    Params      = namedtuple('Params',      ['λ', 'μ', 'σ', 'γ'])
    Params.__qualname__ = "IntermittentExponential.Params" # This is needed for pickle to work.
    HyperParams = namedtuple('HyperParams', ['σ_penalty', 'γ_pr_mean', 'γ_pr_strength'])
    HyperParams.__qualname__ = "IntermittentExponential.HyperParams" # This is needed for pickle to work.
    
    def cdf(self, x, params = None):
        if params is None: params = self.params
        λ, μ, σ, γ = params.λ, params.μ, params.σ, params.γ
        R = fpt.alaplace_cdf(2*λ, 2*μ, x)
        H = norm_dist.cdf(x, scale=σ) if σ > 0 else 0 * R
        return γ*R + (1-γ)*H

    @staticmethod
    def gen_data(M, params):
        λ, μ, σ, γ = params.λ, params.μ, params.σ, params.γ
        y = np.zeros(M)
        z = rand(M) < γ
        i0= ~z
        n0= np.sum(~z)
        i1= z
        n1= np.sum(z)
        pp= λ/(λ + μ)
        ip= rand(M)<pp
        # Set y at ip to be sampled from an exponential with mean λ
        y[ip] = np.random.exponential(λ, sum(ip))
        # Set y at ~ip to be sampled from an exponential with mean μ
        y[~ip]= -np.random.exponential(μ, sum(~ip))       
        y[~z] = randn(n0)*σ
        
        labs             = 0*z
        labs[z & (y>=0)] = 1
        labs[z & (y<0)]  = -1
        
        return y, labs

    def __init__(self, name = "Model", init_params = None, intermittent = True, min_μ = 1e-6, σ_penalty=0, γ_pr_mean=0.5, γ_pr_strength=0):
        INFO("*"*80)
        INFO(f"Initializing model {self.__class__.__name__} named '{name}' {id(self)=}.")
        self.name  = name
        self.min_μ = min_μ
        self.hyper_params= self.HyperParams(σ_penalty=σ_penalty, γ_pr_mean=γ_pr_mean, γ_pr_strength=γ_pr_strength)

        if init_params is not None and not isinstance(init_params, self.Params):
            self.cast_params(init_params)
        else:
            self.init_params = init_params
            self.params      = init_params
            
        self.intermittent = intermittent
        if not self.intermittent:
            if self.init_params is not None:
                self.init_params  = self.init_params._replace(γ = 1)
                self.params       = self.params._replace(γ = 1)
            self.hyper_params = self.hyper_params._replace(γ_pr_mean = 1, γ_pr_strength = np.inf)
        
        INFO(f"Initialized self.init_params = {params2str(self.init_params, self.Params._fields) if self.init_params else self.init_params}")
        INFO(f"Initialized self.params      = {params2str(self.params, self.Params._fields) if self.params else self.params}")
        INFO(f"Initialized self.hyper_params= {params2str(self.hyper_params, self.HyperParams._fields)}")
    def cast_params(self, params):
        γ = params.γ
        σ = params.σ
        if hasfield(params, "α"): # It's a GenInvGauss
            INFO("Casting IntermittentGenInvGauss to IntermittentExponential.")
            λ = 2 * params.λ / params.α
            μ = 2 * params.μ / params.β
        elif hasfield(params, "k"): # It's a Gamma
            INFO("Casting IntermittentGamma to IntermittentExponential.")
            λ = params.λ
            μ = params.μ
        elif hasfield(params, "λ"): # It's an Exponential
            INFO("Casting IntermittentExponential to IntermittentExponential.")
            λ = params.λ
            μ = params.μ        
        else:
            raise ValueError("Unknown params type.")

        self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ)
        self.init_params = self.params
        INFO(f"Cast params: {params2str(params, params._fields)} -> {params2str(self.params, self.Params._fields)}")

    def init_from_fit(self, y, γ = None):
        if γ == None:
            if self.params is None:
                γ = self.hyper_params.γ_pr_mean
            else:
                γ = self.params.γ
                
        INFO(f"Initializing from fit using {γ=:g}.")
        M = len(y)
        ind_z = np.argsort(-abs(y))[:int(M * γ)]    
        z  = 0*y
        z[ind_z] = 1
        z  =  z.astype(bool)
        yp =  y[z & (y>0)]
        yn = -y[z & (y<0)]

        INFO(f"Found {M - sum(z):d} intermittent points, {len(yp):d} positive, {len(yn):d} negative values.")
        # Fit a gamma distribution to the positive values using scipy.stats.gamma.fit
        _, λ = expon_dist.fit(yp, floc=0)
        INFO(f"Fit exponential distribution to positive values, λ={λ:.3g}.")
        if len(yn) == 0:
            μ = self.min_μ
            INFO(f"No negative values, setting μ={μ:.3g}.")
        else:
            _, μ = expon_dist.fit(yn, floc=0)
            INFO(f"Fit exponential distribution to negative values, μ={μ:.3g}.")

        σ = max(np.std(y[~z]),1e-6)
        self.init_params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ)
        self.params      = self.init_params
        INFO(f"Parameters initialized to {params2str(self.params, self.Params._fields)}.")
            
        
    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6, **kwargs):
        INFO(f"Fitting {self.__class__.__name__} model {self.name:s}.")
        assert y is None, "y must be None"

        y = X.flatten()

        # Initialize
        M = len(y)

        if self.init_params is None:
            self.init_from_fit(y)
            
        λ_old, μ_old, σ_old, γ_old = self.init_params.λ, self.init_params.μ, self.init_params.σ, self.init_params.γ
        
        # Take the top M γ active values as those that active
        ind_z = np.argsort(-abs(y))[:int(M * γ_old)]    
        z = 0*y
        z[ind_z] = 1
        z = z.astype(bool)    
        for i in range(max_iter):
            # E-step
            if i > 0:
                if self.intermittent:
                    lrho = 0 * y
                    lrho[y>=0] = -np.log(λ + μ)-abs(y[y>=0])/(λ+1e-8)
                    lrho[y<0]  = -np.log(λ + μ)-abs(y[y< 0])/(μ+1e-8)
                    leta       = -y**2/2/σ**2 - np.log(np.sqrt(2*np.pi*σ**2))
                    z          = (lrho - leta) > np.log((1-min(γ,1-1e-8))/γ)
                else:
                    z = (0*y + 1).astype(bool)
            # M-step
            i0= ~z
            n0= sum(i0)        
            
            if self.intermittent:
                γ = (sum(z) + self.hyper_params.γ_pr_mean*self.hyper_params.γ_pr_strength*len(z))/(len(z) + self.hyper_params.γ_pr_strength*len(z))
                
                if n0>0:
                    y0 = y[i0]
                    if self.hyper_params.σ_penalty > 0: σ2 = n0/(2*self.hyper_params.σ_penalty) * (-1 + np.sqrt(1 + 4 * self.hyper_params.σ_penalty * np.mean(y0**2)/n0))
                    else:                               σ2 = np.mean(y0**2)
                else:
                    σ2 = np.var(y)*1e-3 # Don't make it exactly 0
            else:
                γ  = 1
                σ2 = 0
    
            σ = np.sqrt(σ2)
                        
            ip  = z & (y>=0)
            i_n = z & (y<0)    
            n_p = sum(ip)
            nn  = sum(i_n)
            n   = n_p + nn
            ypm = np.mean(y[ip]) if n_p else 0
            ynm = np.mean(abs(y[i_n])) if nn else 0
            λ   = 1/(n+1e-8) * (n_p*ypm + np.sqrt(n_p * nn * ypm * ynm))
            μ   = max(1/(n+1e-8) * (nn*ynm + np.sqrt(n_p * nn * ypm * ynm)), self.min_μ)        

            self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ)
            # Print the values at the current iteration, including the iteration number
            DEBUG(f"Iter {i:>4d}: n+={sum(ip):>4d}, n-={sum(i_n):>4d} n0={n0:>4d} " + params2str(self.params, self.Params._fields))
            # Check convergence
            if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ - σ_old) < tol and abs(γ - γ_old) < tol:
                INFO(f"Converged in {i:>4d} iterations to n+={sum(ip):>4d}, n-={sum(i_n):>4d} n0={n0:>4d} " + params2str(self.params, self.Params._fields))
                break
            λ_old = λ
            μ_old = μ
            σ_old = σ
            γ_old = γ
    
        labs = 0*z
        labs[z & (y>=0)] = 1
        labs[z & (y<0)]  = -1
        self.labs = labs
        
class IntermittentGamma(IntermittentExponential):
    Params      = namedtuple('Params',      ['λ', 'μ', 'σ', 'γ', 'k', 'm'])
    Params.__qualname__ = 'IntermittentGamma.Params'
    HyperParams = namedtuple('HyperParams', ['σ_penalty', 'γ_pr_mean', 'γ_pr_strength'])
    HyperParams.__qualname__ = 'IntermittentGamma.HyperParams'

    @staticmethod
    def gam_Z(λ, k):
        return gamma_fun(k) * (λ**k)
    
    @staticmethod
    def agam_Z(params):        
        λ = params.λ
        μ = params.μ
        k = params.k
        m = params.m

        Z0 = IntermittentGamma.gam_Z(μ, m)
        Z1 = IntermittentGamma.gam_Z(λ, k)
        Z  = Z0 + Z1

        return Z, Z0, Z1
    
    @staticmethod
    def agamma_cdf(y, params):
        λ = params.λ
        μ = params.μ
        k = params.k
        m = params.m

        n_neg = sum(y<0)
        n_pos = sum(y>=0)

        if n_neg == 0:
            DEBUG(f"agamma_cdf: n_neg = 0 so setting Z0 = 0")
            Z0 = 0
        else:
            Z0 = gamma_fun(m) * (μ**m)

        Z1 = gamma_fun(k) * (λ**k)
        DEBUG(f"agamma_cdf: Z0 = {Z0}, Z1 = {Z1}")
        Z  =  Z0 + Z1
        

        cdf = 0*y
        cdf[y<0]  = (1 -   gamma_dist.cdf(abs(y[y<0]), m, scale = μ)) * Z0/Z
        cdf[y>=0] = Z0/Z + gamma_dist.cdf(y[y>=0],     k, scale = λ) * Z1/Z
        return cdf

    def cdf(self, y, params = None):
        if params is None: params = self.params
        # Compute the CDF
        R = self.agamma_cdf(y, params)
        H = norm_dist.cdf(y, scale = params.σ) if self.intermittent else 0*R
        return params.γ*R + (1-params.γ)*H
    
    @staticmethod
    def gen_data(M, params):
        λ, μ, σ, γ, k, m = [params._asdict()[k] for k in ['λ', 'μ', 'σ', 'γ', 'k', 'm']]
        y = np.zeros(M)
        z = rand(M) < γ
        i0= ~z
        n0= np.sum(~z)
        i1= z
        n1= np.sum(z)
        pp= λ**k * gamma_fun(k)/(λ**k * gamma_fun(k) + μ**m * gamma_fun(m))
        ip= rand(M)<pp
        # Set y at ip to be sampled from an exponential with mean λ
        y[ip] = np.random.gamma(k, λ, sum(ip))
        # Set y at ~ip to be sampled from an exponential with mean μ
        y[~ip]= -np.random.gamma(m, μ, sum(~ip))       
        y[~z] = randn(n0)*σ
        
        labs = 0*z
        labs[z & (y>=0)] = 1
        labs[z & (y<0)]  = -1

        return y, labs

    @staticmethod
    def params_from_gig(gig_params):
        λ = 2*gig_params.λ / gig_params.α
        μ = 2*gig_params.μ / gig_params.β
        k = gig_params.k
        m = gig_params.m
        σ = gig_params.σ
        γ = gig_params.γ
        return IntermittentGamma.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m)        

    @staticmethod
    def params_from_exp(exp_params):
        λ = exp_params.λ
        μ = exp_params.μ
        k = 1
        m = 1
        σ = exp_params.σ
        γ = exp_params.γ
        return IntermittentGamma.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m)        
    
    @staticmethod
    def neg_ll(p, fp, fn, ypm, ynm, lypm, lynm):
        λ, μ, k, m = p
        lZ   = np.log(gamma_fun(m) * (μ**m) + gamma_fun(k) * (λ**k))
        lpos = fp*(-(k-1)*lypm + ypm/λ)
        lneg = fn*(-(m-1)*lynm + ynm/μ)
        return (lZ + lpos + lneg)

    def init_from_fit(self, y, γ = None):
        if γ == None:
            if self.params is None:
                γ = self.hyper_params.γ_pr_mean
            else:
                γ = self.params.γ

        INFO(f"Initializing from fit using {γ=:g}.")
        M = len(y)
        ind_z = np.argsort(-abs(y))[:int(M * γ)]    
        z  = 0*y
        z[ind_z] = 1
        z  =  z.astype(bool)
        yp =  y[z & (y>0)]
        yn = -y[z & (y<0)]
        INFO(f"Found {M - sum(z):d} intermittent points, {len(yp):d} positive, {len(yn):d} negative values.")
        # Fit a gamma distribution to the positive values using scipy.stats.gamma.fit
        k, _, λ = gamma_dist.fit(yp, floc=0)
        INFO(f"Fit gamma distribution to positive values, k={k:.3g}, λ={λ:.3g}.")
        m, _, μ = gamma_dist.fit(yn, floc=0)
        INFO(f"Fit gamma distribution to negative values, m={m:.3g}, μ={μ:.3g}.")

        σ = max(np.std(y[~z]),1e-6)

        self.init_params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m)
        self.params      = self.init_params
        INFO(f"Parameters initialized to {params2str(self.params, self.Params._fields)}.")

    def cast_params(self, params):
        γ = params.γ
        σ = params.σ
        if hasfield(params, "α"): # It's a GenInvGauss
            INFO("Casting IntermittentGenInvGauss to IntermittentGamma.")
            λ = max(1e-6, 2 * params.λ / params.α)
            μ = max(1e-6, 2 * params.μ / params.β)
            k = max(1e-6, params.k)
            m = max(1e-6, params.m)
            λ, k = IntermittentGeneralizedInverseGaussian.nearest_gamma(params.λ, params.k, params.α, x0 = np.array([λ, k]))
            μ, m = IntermittentGeneralizedInverseGaussian.nearest_gamma(params.μ, params.m, params.β, x0 = np.array([μ, m]))            
        elif hasfield(params, "k"): # It's a Gamma
            INFO("Casting IntermittentGamma to IntermittentGamma.")
            λ = params.λ
            μ = params.μ
            k = params.k
            m = params.m
        elif not hasfield(params, "k"): # It's an Exponential
            INFO("Casting IntermittentExponential to IntermittentGamma.")
            λ = params.λ
            μ = params.μ
            k = 1
            m = 1
        else:
            raise ValueError("Casting from unknown params type.")

        self.params      = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m)
        self.init_params = self.params        
        INFO(f"Cast params: {params2str(params, params._fields)} -> {params2str(self.params, self.Params._fields)}")
        
    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6, damping = 0.5, max_fp_iter=1000, method = "Nelder-Mead", **kwargs):
        INFO(f"Fitting {self.__class__.__name__} model {self.name}.")
        assert y is None, "y must be None"
        y = X.flatten()

        # Initialize
        M     = len(y)

        # Initialize the parameters from self.init_params
        if self.init_params is None:
            self.init_from_fit(y)
            
        λ, μ, σ, γ, k, m = [self.init_params._asdict()[k] for k in ['λ', 'μ', 'σ', 'γ', 'k', 'm']]    
        λ_old, μ_old, σ_old, γ_old, k_old, m_old = λ, μ, σ, γ, k, m
    
        ZFUN = lambda λ, μ, k, m: gamma_fun(m) * μ**m + gamma_fun(k) * λ**k
        for i in range(max_iter):
            Z = ZFUN(λ, μ, k, m)
    
            # E-step
            if self.intermittent:
                lrho       = 0 * y
                lrho[y>=0] = -np.log(Z) + (k - 1)*np.log(y[y>=0]) - y[y>=0]/(λ+1e-8)
                lrho[y<0]  = -np.log(Z) + (m - 1)*np.log(-y[y<0]) - abs(y[y<0])/(μ+1e-8)
                leta       = -(y**2)/2/σ**2 - np.log(np.sqrt(2*np.pi*σ**2))
                z          = (lrho - leta) > np.log((1-min(γ,1-1e-8))/γ)
            else:
                z = (0*y + 1).astype(bool)

            i0= ~z
            n0= sum(i0)        
            
            # M-step
            if self.intermittent:
                γ = (sum(z) + self.hyper_params.γ_pr_mean*self.hyper_params.γ_pr_strength*len(z))/(len(z) + self.hyper_params.γ_pr_strength*len(z))
        
                if n0>0:
                    y0 = y[i0]
                    if self.hyper_params.σ_penalty>0: σ2 = n0/(2*self.hyper_params.σ_penalty) * (-1 + np.sqrt(1 + 4 * self.hyper_params.σ_penalty * np.mean(y0**2)/n0))
                    else:                             σ2 = np.mean(y0**2)
                else:
                    σ2 = np.var(y)*1e-3 # Don't make it exactly 0
            else:
                γ = 1
                σ2= 0
    
            σ = np.sqrt(σ2)
                        
            ip   = z & (y>=0)
            i_n  = z & (y<0)    
            n_p  = sum(ip)
            nn   = sum(i_n)
            n    = n_p + nn
            ypm  = np.mean(y[ip]) if n_p else 0
            lypm = np.mean(np.log(y[ip])) if n_p else 0
            ynm  = np.mean(abs(y[i_n])) if nn else 0
            lynm = np.mean(np.log(abs(y[i_n]))) if nn else 0

            fixed_point_function = lambda x: np.array([
                (ZFUN(x[0], x[2], x[1], x[3])/n/gamma_fun(x[1])/x[1]*ypm*n_p)**(1/(x[1]+1)),
                (np.log(x[0]) + digamma(x[1]))/x[0]*ypm/lypm,
                self.min_μ if nn==0 else (ZFUN(x[0], x[2], x[1], x[3])/n/gamma_fun(x[3])/x[3]*ynm*nn)**(1/(x[3]+1)),
                1 if nn==0 else (np.log(x[2]) + digamma(x[3]))/x[2]*ynm/lynm
            ])

            if method == "FP":
                fixed_point_init = np.array([λ_old, k_old, μ_old, m_old])
                
                (λ, k, μ, m), status = fixed_point_iterate(fixed_point_function,
                                                           fixed_point_init,
                                                           damping  = damping,
                                                           max_iter = max_fp_iter,)
            else:
                bnds = [(1e-6, 10)]*4
                x0 = np.array([λ_old, μ_old, k_old, m_old])
                x0 = np.array([np.clip(x0[i], bnds[i][0], bnds[i][1]) for i in range(len(x0))])
                sol = minimize(self.neg_ll, x0,
                           args   = (n_p/(n+1e-8), nn/(n+1e-8), ypm, ynm, lypm, lynm),
                            bounds = bnds,
                           method = method)
                λ, μ, k, m = sol.x
                
            # Print the values at the current iteration, including the iteration number
            self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m)
            DEBUG(f"Iter {i:>4d}: n+={n_p:>4d}, n-={nn:>4d} n0={len(ip) -nn - n_p:>4d} " + params2str(self.params, self.Params._fields))
            # Check convergence
            if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ - σ_old) < tol and abs(γ - γ_old) < tol and abs(m - m_old) < tol and abs(k - k_old) < tol:
                DEBUG(f"Converged in {i:>4d} iterations to n+={n_p:>4d}, n-={nn:>4d} n0={len(ip) -nn - n_p:>4d} " + params2str(self.params, self.Params._fields))
                break
            λ_old = λ
            μ_old = μ
            σ_old = σ
            γ_old = γ
            m_old = m
            k_old = k
    
        labs = 0*z
        labs[z & (y>=0)] = 1
        labs[z & (y<0)]  = -1
        self.labs = labs

        return self


class IntermittentGeneralizedInverseGaussian(IntermittentExponential):
    Params      = namedtuple('Params',      ['λ', 'μ', 'σ', 'γ', 'k', 'm', 'α', 'β'])
    Params.__qualname__ = 'IntermittentGeneralizedInverseGaussian.Params'
    HyperParams = namedtuple('HyperParams', ['σ_penalty', 'γ_pr_mean', 'γ_pr_strength'])
    HyperParams.__qualname__ = 'IntermittentGeneralizedInverseGaussian.HyperParams'

    @staticmethod
    def gig_Z(η, p, θ):
        return 2 * η * kv(p, θ)

    @staticmethod
    def nearest_gamma(η, p, θ, x0 = None, method="TNC",  **kwargs):
        """Project a GIG distribution to a Gamma distribution
        by computing the M-projection. That is, minimizing
        D(P||Q) where P is the GIG distribution and Q is the
        Gamma distribution. Since
        
        D(P||Q) = E_P[log(P/Q)] = E_P[log(P)] - E_P[log(Q)]

        we need only maximize E_P[log(Q)].

        Q(x) = x^(k-1) exp(-x/θ) / Γ(k) θ^k, so
        
        log(Q(x)) = (k-1) log(x) - x/θ - log(Γ(k)) - k log(θ)

        E_P[log(Q)] = (k-1) E_P[log(x)] - E_P[x]/θ - log(Γ(k)) - k log(θ)
                    = (k-1) C_1 - C_0/θ - log(Γ(k)) - k log(θ)

        d/dθ E_P[log(Q)] = C_0/θ^2 - k/θ
                    = 0 => k = C_0/θ 

        d/dk E_P[log(Q)] = (k-1) C_1 - Γ'(k)/Γ(k) - log(θ)
                    = 0 => θ = Exp((k-1)C_1 - Γ'(k)/Γ(k))
        """
        INFO(f"Finding nearest Gamma distribution to GIG({η=}, {p=}, {θ=}).")
        C0= η * kv(p+1, θ)/kv(p, θ) #E_P[x] = sqrt(b/a) K_{p+1}(sqrt(ab))/K_p(sqrt(ab))
        C1= np.log(η) + kvv(p, θ)/kv(p,θ) # E_P(ln x) = ln(sqrt(b/a)) + d/dp ln K_p(sqrt(ab))
        DEBUG(f"{C0=}, {C1=}")

        if x0 is None:
            x0 = np.clip(np.array([2 * η / θ, p]), 1e-6, 10)
        DEBUG(f"Initializing at {x0}.")            
        
        if method == "FP":
            DEBUG("Solving using fixed point iteration.")
            fθk = lambda p: np.array([C0/p[0], np.exp((p[1]-1)*C1 - digamma(p[1])/gamma_fun(p[1]))])
            sol, status = fixed_point_iterate(fkθ, x0, **kwargs)
            DEBUG(f"Fixed point iteration terminated with {status=}.")
            θ, k = sol
        else:
            DEBUG("Solving using scipy.optimize.minimize.")
            neg_ElogQ = lambda p: -((p[1]-1)*C1 - C0/p[0] - np.log(gamma_fun(p[1])) - p[1]*np.log(p[0]))
            DEBUG(f"Initial value of neg_ElogQ = {neg_ElogQ(x0)}.")
            sol = minimize(neg_ElogQ, x0,
                           bounds = [(1e-6, 100), (1e-6, 100)],
                           method=method, **kwargs)
            θ, k = sol.x
            DEBUG(f"Minimization terminated with {sol.message}.")
            DEBUG(f"Final value of neg_ElogQ = {neg_ElogQ(sol.x)}.")
            
            
        INFO(f"Mapped to {θ=}, {k=}.")
        return θ, k

    
        


    @staticmethod
    def agig_Z(params):        
        λ = params.λ
        μ = params.μ
        k = params.k
        m = params.m
        α = params.α
        β = params.β

        Z0 = IntermittentGeneralizedInverseGaussian.gig_Z(μ, m, β)        
        Z1 = IntermittentGeneralizedInverseGaussian.gig_Z(λ, k, α)
        Z  = Z0 + Z1

        return Z, Z0, Z1

    @staticmethod
    def agig_cdf(y, params, gammaness_thresh):
        λ = params.λ
        μ = params.μ
        k = params.k
        m = params.m
        α = params.α
        β = params.β

        n_neg = sum(y<0)
        n_pos = sum(y>=0)
        
        gammaness = [IntermittentGeneralizedInverseGaussian.gammaness(p) for p in [μ, λ]]
        if gammaness[0] > gammaness_thresh:
            gamma_neg_scale = 2*μ/β            
            INFO(f"Using Gamma distribution with scale = {gamma_neg_scale:.2g} for negative values, since gammaness {gammaness[0]:.1e} > {gammaness_thresh:.1e}.")
            if n_neg > 0:
                Z0 = IntermittentGamma.gam_Z(gamma_neg_scale, m)
            else:
                Z0 = 0
                DEBUG("No negative values, so Z0 = 0.")
            neg_cdf = lambda y: (1 - gamma_dist.cdf(np.abs(y), m, scale=gamma_neg_scale))
        else:
            Z0 = IntermittentGeneralizedInverseGaussian.gig_Z(μ, m, β)
            neg_cdf = lambda y: (1 - geninvgauss.cdf(np.abs(y), m, β, scale = μ))

        if gammaness[1] > gammaness_thresh:
            gamma_pos_scale = 2*λ/α                      
            INFO(f"Using Gamma distribution with scale = {gamma_pos_scale:.2g} for positive values, since gammaness {gammaness[1]:.1e} > {gammaness_thresh:.1e}.")
            Z1 = IntermittentGamma.gam_Z(gamma_pos_scale, k)
            pos_cdf = lambda y: gamma_dist.cdf(y, k, scale=gamma_pos_scale)
        else:
            Z1 = IntermittentGeneralizedInverseGaussian.gig_Z(λ, k, α)
            pos_cdf = lambda y: geninvgauss.cdf(y, k, α, scale = λ)

        Z         = Z0 + Z1
        cdf       = 0*y
        cdf[y<0]  = neg_cdf(y[y<0]) * Z0/Z
        cdf[y>=0] = Z0/Z + pos_cdf(y[y>=0]) * Z1/Z

        out_of_range = (cdf < 0) | (cdf > 1)
        if np.any(out_of_range):
            WARN(f"cdf out of range for {np.sum(out_of_range)} of {len(out_of_range)} values.")
            cdf[out_of_range] = np.nan
        return cdf

    def cdf(self, y, params = None):
        if params is None: params = self.params
        # Compute the CDF
        R = self.agig_cdf(y, params, self.gammaness_thresh)
        H = norm_dist.cdf(y, scale = params.σ) if self.intermittent else 0*R
        return params.γ*R + (1-params.γ)*H
    
    @staticmethod
    def gen_data(M, params):
        λ, μ, σ, γ, k, m, α, β = [params._asdict()[k] for k in ['λ', 'μ', 'σ', 'γ', 'k', 'm', 'α', 'β']]
        y = np.zeros(M)
        z = rand(M) < γ
        i0= ~z
        n0= np.sum(~z)
        i1= z
        n1= np.sum(z)
        Z, Z0, Z1 = IntermittentGeneralizedInverseGaussian.agig_Z(params)
        pp= Z1/(Z1 + Z0)
        ip= rand(M)<pp
        y[ip]  =  geninvgauss.rvs(k, α, scale=λ, size=sum(ip))
        y[~ip] = -geninvgauss.rvs(m, β, scale=μ, size=sum(~ip))       
        y[~z]  =  randn(n0)*σ
        
        labs             = 0*z
        labs[z & (y>=0)] = 1
        labs[z & (y<0)]  = -1

        return y, labs

    @staticmethod
    def neg_ll(p, fp, fn, ypm, ynm, iypm, iynm, lypm, lynm):
        λ, μ, k, m, α, β = p
        lZ   = np.log(2*(λ * kv(k, α) + μ * kv(m, β)))
        lpos = fp*((k-1)*np.log(λ) - (k-1)*lypm + α/(2*λ) * ypm + α * λ/2 * iypm)
        lneg = fn*((m-1)*np.log(μ) - (m-1)*lynm + β/(2*μ) * ynm + β * μ/2 * iynm)
        return (lZ + lpos + lneg)

    @staticmethod
    def gammaness(λ):
        # Measures how close the data is to the gamma distribution
        # For an IGIG distribution, the exponential term is exp(-α(x/λ + λ/x)/2)
        # The gamma distribution has the exponential term exp(-x/λ)
        # So the gammaness is the ratio of the two exponents in th IGIG term.
        # If it's very large, then the data is very gamma-like, since the 1/x term is small.
        return λ**-2

    def __init__(self, *args, gammaness_thresh = np.inf, **kwargs):
        # Call the parent constructor
        super().__init__(*args, **kwargs)
        self.gammaness_thresh = gammaness_thresh # How gamma-like the data must be to be considered gamma distributed.

    def init_from_fit(self, y, γ = None):
        if γ == None:
            if self.params is None:
                γ = self.hyper_params.γ_pr_mean
            else:
                γ = self.params.γ

        INFO(f"Initializing from fit using {γ=:g}.")
        M = len(y)
        ind_z = np.argsort(-abs(y))[:int(M * γ)]    
        z  = 0*y
        z[ind_z] = 1
        z  =  z.astype(bool)
        yp =  y[z & (y>0)]
        yn = -y[z & (y<0)]
        INFO(f"Found {M - sum(z):d} intermittent points, {len(yp):d} positive, {len(yn):d} negative values.")
        # Fit a gamma distribution to the positive values using scipy.stats.gamma.fit
        k, α, _, λ = geninvgauss.fit(yp, floc=0)
        INFO(f"Fit geninvgauss distribution to positive values, {λ=:.3g}, {k=:.3g}, {α=:.3g}.")
        m, β, _, μ = geninvgauss.fit(yn, floc=0)
        INFO(f"Fit geninvgauss distribution to negative values, {m=:.3g}, {μ=:.3g}, {β=:.3g}.")

        σ = max(np.std(y[~z]),1e-6)

        self.init_params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m, α=α, β=β)
        self.params      = self.init_params
        INFO(f"Parameters initialized to {params2str(self.params, self.Params._fields)}.")

    def cast_params(self, params):
        γ = params.γ
        σ = params.σ
        if hasfield(params, "α"): # It's an IGIG
            INFO("Casting IntermittentGeneralizedInversGaussian to IntermittentGeneralizedInversGaussian.")
            α = params.α
            β = params.β
            λ = params.λ
            μ = params.μ
            k = params.k
            m = params.m
        elif hasfield(params, "k"): # It's a Gamma
            INFO("Casting IntermittentGamma to IntermittentGeneralizedInversGaussian.")
            # 2 λ α = 1e-6 # To make it gamma            
            # 2 λ / α = scale
            # scale * α**2 = 2 λ/α * α**2 = 2 λ α = 1e-6
            α = np.sqrt(1e-6/params.λ)
            β = np.sqrt(1e-6/params.μ)
            λ = α * params.λ /2
            μ = β * params.μ / 2
            k = params.k
            m = params.m
        elif not hasfield(params, "k"): # It's an Exponential
            INFO("Casting IntermittentExponential to IntermittentGeneralizedInversGaussian.")
            α = np.sqrt(1e-6/params.λ)
            β = np.sqrt(1e-6/params.μ)
            λ = α * params.λ /2
            μ = β * params.μ / 2
            k = 1
            m = 1
        else:
            raise ValueError("Casting from unknown params type.")

        self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m, α = α, β = β)
        self.init_params = self.params        
        INFO(f"Cast params: {params2str(params, params._fields)} -> {params2str(self.params, self.Params._fields)}")

        
    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6, damping = 0.5, max_fp_iter=1000, method = "Nelder-Mead"):
        INFO(f"Fitting IntermittentGeneralizedInverseGaussian model {self.name}.")
        assert y is None, "y must be None"
        y = X.flatten()

        # Initialize
        M = len(y)

        if self.params is None:
            self.init_from_fit(y)
            
        # Initialize the parameters from self.init_params    
        λ, μ, σ, γ, k, m, α, β = [self.init_params._asdict()[k] for k in ['λ', 'μ', 'σ', 'γ', 'k', 'm', 'α', 'β']]    
        λ_old, μ_old, σ_old, γ_old, k_old, m_old, α_old, β_old = λ, μ, σ, γ, k, m, α, β
    
        for i in range(max_iter):
            Z = self.agig_Z(self.params)[0]
    
            # E-step
            if self.intermittent:
                lrho       = 0 * y
                lrho[y>0]  = -np.log(Z) + (k - 1) * np.log( y[y>0]/λ) - α/2*(y[y>0]/λ + λ/y[y>0])
                lrho[y<0]  = -np.log(Z) + (m - 1) * np.log(-y[y<0]/μ) + β/2*(y[y<0]/μ + μ/y[y<0])
                leta       = -(y**2)/2/σ**2 - np.log(np.sqrt(2*np.pi*σ**2))
                z          = (lrho - leta) > np.log((1-min(γ,1-1e-8))/γ)
            else:
                z = (0*y + 1).astype(bool)

            i0= ~z
            n0= sum(i0)        
            
            # M-step
            if self.intermittent:
                γ = (sum(z) + self.hyper_params.γ_pr_mean*self.hyper_params.γ_pr_strength*len(z))/(len(z) + self.hyper_params.γ_pr_strength*len(z))
        
                if n0>0:
                    y0 = y[i0]
                    if self.hyper_params.σ_penalty>0: σ2 = n0/(2*self.hyper_params.σ_penalty) * (-1 + np.sqrt(1 + 4 * self.hyper_params.σ_penalty * np.mean(y0**2)/n0))
                    else:                             σ2 = np.mean(y0**2)
                else:
                    σ2 = np.var(y)*1e-3 # Don't make it exactly 0
            else:
                γ = 1
                σ2= 0
    
            σ = np.sqrt(σ2)
                        
            ip   = z & (y>=0)
            i_n  = z & (y<0)    
            n_p  = sum(ip)
            nn   = sum(i_n)
            n    = n_p + nn

            ypm  = np.mean(y[ip])           if n_p else 0
            ynm  = np.mean(-y[i_n])         if nn  else 0
            lypm = np.mean(np.log(y[ip]))   if n_p else 0
            lynm = np.mean(np.log(-y[i_n])) if nn  else 0
            iypm = np.mean(1/y[ip])         if n_p else 0
            iynm = np.mean(-1/y[i_n])       if nn  else 0

            x0  = np.array([λ_old, μ_old, k_old, m_old, α_old, β_old])
            bnds= [(1e-6, 10)]*2 + [(-10,10)]*2 + [(1e-6, 10)]*2
            x0 = np.array([np.clip(x0[i], bnds[i][0], bnds[i][1]) for i in range(len(x0))])
            
            DEBUG(f"Starting optimization with x0 = " + ", ".join([f"{x0[i]:.3g}" for i in range(len(x0))]))
            sol = minimize(self.neg_ll, x0,
                           args   = (n_p/(n+1e-8), nn/(n+1e-8), ypm, ynm, iypm, iynm, lypm, lynm),
                           bounds = bnds,
                           method = method)
            
            DEBUG(f"{sol.message} after {sol.nit} iterations.")
            λ, μ, k, m, α, β = (1 - damping) * sol.x + damping * np.array([λ_old, μ_old, k_old, m_old, α_old, β_old])
            
            # Print the values at the current iteration, including the iteration number
            self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m, α=α, β=β)
            DEBUG(f"Iter {i:>4d}: n+={n_p:>4d}, n-={nn:>4d} n0={len(ip) -nn - n_p:>4d} " + params2str(self.params, self.Params._fields))

            # Check convergence
            if np.allclose([λ, μ, σ, γ, k, m, α, β], [λ_old, μ_old, σ_old, γ_old, k_old, m_old, α_old, β_old], atol=1e-6, rtol=0):
                DEBUG(f"Converged in {i:>4d} iterations to n+={n_p:>4d}, n-={nn:>4d} n0={len(ip) -nn - n_p:>4d} " + params2str(self.params, self.Params._fields))
                break
            
            λ_old = λ
            μ_old = μ
            σ_old = σ
            γ_old = γ
            m_old = m
            k_old = k
            α_old = α
            β_old = β
    
        labs = 0*z
        labs[z & (y>=0)] = 1
        labs[z & (y<0)]  = -1
        self.labs = labs

        return self


class CorrModel(BaseEstimator): # A wrapper over the correlation models that will aid model selection.
    def __init__(self, name = "Model", scoring="r2", model = "gig", init_params=None, init_mode = None, intermittent = True, γ_pr_mean = 0.5, γ_pr_strength = 1, σ_penalty = 0):
        self.name = name
        self.model = model
        self.init_mode = init_mode
        self.intermittent = intermittent
        self.γ_pr_mean = γ_pr_mean
        self.γ_pr_strength = γ_pr_strength
        self.σ_penalty = σ_penalty
        assert scoring in ["r2", "tv"]
        self.scoring = scoring
        self.init_params = init_params
        
        self.models = []
    
    def fit(self, *args, **kwargs):
        init_params = self.init_params

        self.init_kwargs = {
            "intermittent":self.intermittent,
            "γ_pr_mean":self.γ_pr_mean,
            "γ_pr_strength":self.γ_pr_strength,
            "σ_penalty":self.σ_penalty}
        
        if self.init_mode == "ancestral":
            INFO("")
            INFO(f"Using ancestral initialization for {self.model=}.")
            Models = []
            if self.model in ["gig", "gamma"]:
                INFO("Model is one of Gamma or GIG, so adding exponential as an ancestral model.")
                Models.append(IntermittentExponential)
            if self.model in ["gig"]:
                INFO("Model is GIG, so adding Gamma as an ancestral model.")
                Models.append(IntermittentGamma)
                
            for i, Model in enumerate(Models):
                init_kwargs = self.init_kwargs.copy()
                init_kwargs["name"] = f"{self.name}_{i}"
                model = Model(**init_kwargs, init_params = init_params)
                model.fit(*args, **kwargs)
                init_params = model.params
                self.models.append(model)

        model = {"exp":IntermittentExponential,
                 "gamma":IntermittentGamma,
                 "gig":IntermittentGeneralizedInverseGaussian}[self.model](**self.init_kwargs, init_params = init_params)

        model.fit(*args, **kwargs)
        self.models.append(model)
        return self

    def predict(self, *args, **kwargs):
        return self.models[-1].predict(*args, **kwargs)

    def score(self, *args, **kwargs):
        cmp_fun = {"r2":fpt.r2fun, "tv":fpt.tvfun}[self.scoring]
        return self.models[-1].score(*args, cmp_fun=cmp_fun, **kwargs)
    
    def cdf(self, *args, **kwargs):
        return self.models[-1].cdf(*args, **kwargs)

    def __str__(self):
       return str(self.models[-1]) if len(self.models) else super().__str__()

    def __repr__(self):
       return repr(self.models[-1]) if len(self.models) else super().__repr__()
    

def fit_corrs(y, search_spec):
    s = np.std(y)
    y_= y/s

    # Load the yaml file search_spec
    with open(search_spec, "r") as f:
        search_spec = yaml.load(f, Loader=yaml.FullLoader)
    INFO(f"Loaded search_spec from {search_spec=}")
    # Create a GridSearchCV object with the CorrModel as the estimator
    search = GridSearchCVKeepModels(GridSearchCV(estimator = CorrModel(**search_spec["estimator"]),
                            param_grid = search_spec["param_grid"],
                            cv = ShuffleSplit(**search_spec["cv"]),
                            **search_spec["gridsearch"]))

    # Fit the GridSearchCV object
    search.fit(X=y_, y=None)
    results_df = pd.DataFrame(search.cv_results_)
    filtered_results = results_df[
        [p for p in ['param_model', 'param_intermittent' + 'param_γ_pr_mean', 'mean_test_score', 'std_test_score'] if p in results_df.columns]
    ]
    INFO("\n"+filtered_results.sort_values(by='mean_test_score', ascending=False).to_string(index=False))

    return {"search":search, "scale":s}

class GridSearchCVKeepModels:
    def __init__(self, grid_search_cv):
        self.grid_search = grid_search_cv
        self.fitted_models = {}
        
    def custom_scorer(self, estimator, X, y = None):
        score = estimator.score(X, y)
        params_str = str(estimator.get_params())
        if params_str not in self.fitted_models:
            self.fitted_models[params_str] = []
        self.fitted_models[params_str].append(estimator)
        return score
    
    def fit(self, X, y = None):
        if 'scoring' in self.grid_search.get_params():
            if self.grid_search.get_params()['scoring'] is not None:
                raise ValueError("scoring should not be set in the GridSearchCV object.")
        self.grid_search.set_params(scoring=self.custom_scorer)
        self.grid_search.fit(X, y)
        self.cv_results_ = self.grid_search.cv_results_
        return self

class GridSearchCVKeepModelsWrapper: # Separate them out so we can edit this without having to re-run the above
    def __init__(self, obj):
        self.grid_search   = obj.grid_search
        self.fitted_models = obj.fitted_models
        self.cv_results_   = obj.grid_search.cv_results_

    def _get_indices(self, rank = None, **hyperparams):
        # If hyperparams are provided, use them to get the indices
        if len(hyperparams):
            inds = self._get_indices_by_hyperparams(**hyperparams)
            # The indices are sorted by rank.
            # If rank is provided, use it as th index into the sorted indices.
            if rank is not None:
                assert rank < len(inds), f"rank={rank} is too large for the number of indices matching the hyperparameters ({len(inds)})."
                inds = [inds[rank]]
        else:
            inds = self._get_indices_by_rank(rank)

        return inds
            
    def _get_indices_by_rank(self, rank):
        max_rank      = max(self.cv_results_['rank_test_score'])
        adjusted_rank = (rank % max_rank) + 1
        return [i for i, r in enumerate(self.cv_results_['rank_test_score']) if r == adjusted_rank]
        
    def _get_indices_by_hyperparams(self, **hyperparams):
        # Get the indices matching the hyperparams
        inds = [i for i, p in enumerate(self.cv_results_['params']) if all(k in p and p[k] == v for k, v in hyperparams.items())]
        # Return them sorted by rank
        return sorted(inds, key = lambda i: self.cv_results_['rank_test_score'][i])

    def _get_fitted_models_by_hyperparams(self, **hyperparams):
        fitted_model_key_dicts = {k:eval(k) for k in self.fitted_models.keys()}
        models = [fm for k,fm in self.fitted_models.items() if all(hk in fitted_model_key_dicts[k] and fitted_model_key_dicts[k][hk] == v for hk, v in hyperparams.items())]
        assert len(models) == 1, f"Found {len(models)} models with hyperparams {hyperparams}"
        return models[0]

    def _get_scores(self, score_type, rank=None, **hyperparams):
        indices     = self._get_indices(rank, **hyperparams)        
        scores      = [self.cv_results_[score_type][i] for i in indices]
        hyperparams = [self.cv_results_['params'][i]   for i in indices]
        return scores, hyperparams

    def best_test_score(self, **hyperparams):
        # Returns the best test score for the given hyperparameters
        return self._get_scores('mean_test_score', rank=0, **hyperparams)
    
    def mean_test_score(self, rank=None, **hyperparams):
        return self._get_scores('mean_test_score', rank, **hyperparams)

    def std_test_score(self, rank=None, **hyperparams):
        return self._get_scores('std_test_score', rank, **hyperparams)

    def split_test_score(self, rank=None, **hyperparams):
        indices      = self._get_indices(rank, **hyperparams)        
        split_scores = [[self.cv_results_[f'split{j}_test_score'][i] for j in range(self.grid_search.cv.get_n_splits())] for i in indices]
        hyperparams   = [self.cv_results_['params'][i] for i in indices]
        return split_scores, hyperparams
     
    def hyperparams(self, rank=None, **hp):
        indices = self._get_indices(rank, **hp)
        return [self.cv_results_['params'][i] for i in indices]
    
    def models(self, rank=None, **hp):
        hp_list = self.hyperparams(rank, **hp)
        models  = [self._get_fitted_models_by_hyperparams(**hp) for hp in hp_list]
        return models, hp_list

        
