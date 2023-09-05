# This file contains the various models that can be fit to the data.
import numpy as np
from matplotlib import pylab as plt
from collections import namedtuple
from sklearn.base import BaseEstimator
import logging

from scipy.stats import norm as norm_dist
from scipy.stats import   gamma as gamma_dist
from scipy.special import gamma as gamma_fun
from scipy.special import digamma

import fisher_plumes_tools as fpt
import utils

logger = utils.create_logger("corr_models")
logger.setLevel(logging.DEBUG)
INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

rand = np.random.rand
randn= np.random.randn

params2str  = lambda params: ", ".join([f"{p}={v:<.2g}" for p,v in params._asdict().items()])

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

def plot_cdfs(y, mdls = [], labs = [], figsize=None, n = 1001, gof_fun = fpt.compute_r2_value):
    cdf_true, xv = cdf_data(y)
    plt.plot(xv, cdf_true, "-",label="data" )

    if len(labs) == 0:
        labs = [str(mdl) for mdl in mdls]
        
    for mdl,lab in zip(mdls, labs):
        cdf_mdl = mdl.cdf(xv, mdl.params)
        gof = gof_fun(mdl.predict, y, n = n)
        plt.plot(xv, cdf_mdl, "-", label=f"{lab}: {gof:.3f}")

    plt.xlabel("x")
    plt.ylabel("P(data<x)")
    plt.legend()

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
        WARN("Nans encountered.")
        status = 2
    elif i == max_iter - 1:
        WARN("Not converged.")
        status = 1
    else:
        INFO(f"Converged to fixed point {x} in {i:>4d} iterations.")
        
    return x_new, status       

class Exponential(BaseEstimator):
    Params = namedtuple('Params', ['λ', 'μ'])
    
    @staticmethod
    def cdf(x, params):
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

    def __init__(self, init_params, min_μ = 1e-6):
        self.min_μ = min_μ
        self.init_params = init_params
        self.params      = init_params

    def __repr__(self):
        return f"{self.__class__.__name__}(Params({params2str(self.params)}))"

    def __str__(self):
        return f"{self.__class__.__name__}({params2str(self.params)})"
        
    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6):
        assert y is None, "y must be None"

        y = X.flatten()

        # Initialize
        M     = len(y)
        λ_old, μ_old = self.init_params.λ, self.init_params.μ
    
        for i in range(max_iter):
            ip  = (y>=0)
            i_n = (y<0)    
            n_p = sum(ip)
            nn  = sum(i_n)
            n   = n_p + nn
            ypm = np.mean(y[ip]) if n_p else 0
            ynm = np.mean(abs(y[i_n])) if nn else 0
            λ   = 1/n * (n_p*ypm + np.sqrt(n_p * nn * ypm * ynm))
            μ   = max(1/n * (nn*ynm + np.sqrt(n_p * nn * ypm * ynm)), self.min_μ)        

            self.params = self.Params(λ=λ, μ=μ)
            # Print the values at the current iteration, including the iteration number
            INFO(f"Iter {i:>4d}: n+={sum(ip):>4d}, n-={sum(i_n):>4d} " + params2str(self.params))
            # Check convergence
            if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol:
                INFO("Converged.")
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
    HyperParams = namedtuple('HyperParams', ['σ_penalty', 'γ_pr_mean', 'γ_pr_strength'])
    
    @staticmethod
    def cdf(x, params):
        λ, μ, σ, γ = params.λ, params.μ, params.σ, params.γ
        R = fpt.alaplace_cdf(2*λ, 2*μ, x)
        H = norm_dist.cdf(x, scale=σ)
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

    def __init__(self, init_params, min_μ = 1e-6, σ_penalty=0, γ_pr_mean=0.5, γ_pr_strength=0):
        self.min_μ = min_μ
        self.hyper_params= self.HyperParams(σ_penalty=σ_penalty, γ_pr_mean=γ_pr_mean, γ_pr_strength=γ_pr_strength)
        self.init_params = init_params
        self.params      = init_params

    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6):
        assert y is None, "y must be None"

        y = X.flatten()

        # Initialize
        M     = len(y)
        γ_old = self.hyper_params.γ_pr_mean
        λ_old, μ_old, σ_old = self.init_params.λ, self.init_params.μ, self.init_params.σ

        # Take the top M γ active values as those that active
        ind_z = np.argsort(-abs(y))[:int(M * γ_old)]    
        z = 0*y
        z[ind_z] = 1
        z = z.astype(bool)    
        for i in range(max_iter):
            # E-step
            if i > 0:
                lrho = 0 * y
                lrho[y>=0] = -np.log(λ + μ)-abs(y[y>=0])/λ
                lrho[y<0]  = -np.log(λ + μ)-abs(y[y< 0])/μ
                leta       = -y**2/2/σ**2 - np.log(np.sqrt(2*np.pi*σ**2))
                z          = (lrho - leta) > np.log((1-min(γ,1-1e-8))/γ)
            # M-step
            γ = (sum(z) + self.hyper_params.γ_pr_mean*self.hyper_params.γ_pr_strength*len(z))/(len(z) + self.hyper_params.γ_pr_strength*len(z))
            
            i0= ~z
            n0= sum(i0)        
            if n0>0:
                y0 = y[i0]
                if self.hyper_params.σ_penalty > 0: σ2 = n0/(2*self.hyper_params.σ_penalty) * (-1 + np.sqrt(1 + 4 * self.hyper_params.σ_penalty * np.mean(y0**2)/n0))
                else:                               σ2 = np.mean(y0**2)
            else:
                σ2 = np.var(y)*1e-3 # Don't make it exactly 0

            σ = np.sqrt(σ2)
                        
            ip  = z & (y>=0)
            i_n = z & (y<0)    
            n_p = sum(ip)
            nn  = sum(i_n)
            n   = n_p + nn
            ypm = np.mean(y[ip]) if n_p else 0
            ynm = np.mean(abs(y[i_n])) if nn else 0
            λ   = 1/n * (n_p*ypm + np.sqrt(n_p * nn * ypm * ynm))
            μ   = max(1/n * (nn*ynm + np.sqrt(n_p * nn * ypm * ynm)), self.min_μ)        

            self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ)
            # Print the values at the current iteration, including the iteration number
            INFO(f"Iter {i:>4d}: n+={sum(ip):>4d}, n-={sum(i_n):>4d} n0={n0:>4d} " + params2str(self.params))
            # Check convergence
            if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ - σ_old) < tol and abs(γ - γ_old) < tol:
                INFO("Converged.")
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
    HyperParams = namedtuple('HyperParams', ['σ_penalty', 'γ_pr_mean', 'γ_pr_strength'])
    
    @staticmethod
    def agamma_cdf(y, params):
        λ = params.λ
        μ = params.μ
        k = params.k
        m = params.m
    
        Z0 = gamma_fun(m) * (μ**m)
        Z1 = gamma_fun(k) * (λ**k)
        Z  =  Z0 + Z1
        cdf = 0*y
        cdf[y<0]  = (1 -   gamma_dist.cdf(abs(y[y<0]), m, scale = μ)) * Z0/Z
        cdf[y>=0] = Z0/Z + gamma_dist.cdf(y[y>=0],     k, scale = λ) * Z1/Z
        return cdf

    @staticmethod    
    def cdf(y, params):
        λ, μ, k, m, σ, γ = [params._asdict()[k] for k in ['λ', 'μ', 'k', 'm', 'σ', 'γ']]    
        # Compute the CDF
        R = IntermittentGamma.agamma_cdf(y, params)
        H = norm_dist.cdf(y, scale = σ)
        return γ*R + (1-γ)*H
    
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

    def fit(self, X, y=None, max_iter = 1001, tol = 1e-6, damping = 0.5, max_fp_iter=1000):
        assert y is None, "y must be None"
        y = X.flatten()

        # Initialize
        M     = len(y)

        # Initialize the parameters from self.init_params
        λ, μ, σ, γ, k, m = [self.init_params._asdict()[k] for k in ['λ', 'μ', 'σ', 'γ', 'k', 'm']]    
        λ_old, μ_old, σ_old, γ_old, k_old, m_old = λ, μ, σ, γ, k, m
    
        ZFUN = lambda λ, μ, k, m: gamma_fun(m) * μ**m + gamma_fun(k) * λ**k
        for i in range(max_iter):
            Z = ZFUN(λ, μ, k, m)
    
            # E-step        
            lrho       = 0 * y
            lrho[y>=0] = -np.log(Z) + (k - 1)*np.log(y[y>=0]) - y[y>=0]/λ
            lrho[y<0]  = -np.log(Z) + (m - 1)*np.log(-y[y<0]) - abs(y[y<0])/μ
            leta       = -(y**2)/2/σ**2 - np.log(np.sqrt(2*np.pi*σ**2))
            z          = (lrho - leta) > np.log((1-min(γ,1-1e-8))/γ)
    
            # M-step        
            γ = (sum(z) + self.hyper_params.γ_pr_mean*self.hyper_params.γ_pr_strength*len(z))/(len(z) + self.hyper_params.γ_pr_strength*len(z))
    
            i0= ~z
            n0= sum(i0)        
            if n0>0:
                y0 = y[i0]
                if self.hyper_params.σ_penalty>0: σ2 = n0/(2*self.hyper_params.σ_penalty) * (-1 + np.sqrt(1 + 4 * self.hyper_params.σ_penalty * np.mean(y0**2)/n0))
                else:                             σ2 = np.mean(y0**2)
            else:
                σ2 = np.var(y)*1e-3 # Don't make it exactly 0

            σ = np.sqrt(σ2)
                        
            ip   = z & (y>=0)
            i_n  = z & (y<0)    
            n_p  = sum(ip)
            nn   = sum(i_n)
            n    = n_p + nn
            yps  = sum(y[ip]) if n_p else 0
            lyps = sum(np.log(y[ip])) if n_p else 0
            yns  = sum(abs(y[i_n])) if nn else 0
            lyns = sum(np.log(abs(y[i_n]))) if nn else 0

            fixed_point_function = lambda x: np.array([
                (ZFUN(x[0], x[2], x[1], x[3])/n/gamma_fun(x[1])/x[1]*yps)**(1/(x[1]+1)),
                (np.log(x[0]) + digamma(x[1]))/x[0]*yps/lyps,
                self.min_μ if nn==0 else (ZFUN(x[0], x[2], x[1], x[3])/n/gamma_fun(x[3])/x[3]*yns)**(1/(x[3]+1)),
                1 if nn==0 else (np.log(x[2]) + digamma(x[3]))/x[2]*yns/lyns
            ])

            fixed_point_init = np.array([λ_old, k_old, μ_old, m_old])
            
            (λ, k, μ, m), status = fixed_point_iterate(fixed_point_function,
                                                       fixed_point_init,
                                                       damping  = damping,
                                                       max_iter = max_fp_iter,)
            
            # Print the values at the current iteration, including the iteration number
            self.params = self.Params(λ=λ, μ=μ, σ=σ, γ=γ, k=k, m=m)
            INFO(f"Iter {i:>4d}: n+={n_p:>4d}, n-={nn:>4d} n0={len(ip) -nn - n_p:>4d} " + params2str(self.params))
            # Check convergence
            if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ - σ_old) < tol and abs(γ - γ_old) < tol and abs(m - m_old) < tol and abs(k - k_old) < tol:
                INFO("Converged.")
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
