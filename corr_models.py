# This file contains the various models that can be fit to the data.
import numpy as np
from matplotlib import pylab as plt
from collections import namedtuple
from sklearn.base import BaseEstimator
import logging

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

def plot_cdfs(y, mdls = [], figsize=None, n = 1001, gof_fun = fpt.compute_r2_value):
    cdf_true, xv = cdf_data(y)
    plt.plot(xv, cdf_true, "-",label="data" )

    for mdl in mdls:
        cdf_mdl = mdl.cdf(xv, mdl.params)
        gof = gof_fun(mdl.predict, y, n = n)
        plt.plot(xv, cdf_mdl, "-", label=f"{str(mdl)}: {gof:.3f}")

    plt.xlabel("x")
    plt.ylabel("P(data<x)")
    plt.legend()
    

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
        return f"Exponential(Params({params2str(self.params)}))"

    def __repr__(self):
        return f"Exponential({params2str(self.params)})"
    
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

        return self

    def predict(self, X):
        # Return the cdf at the values in X.
        return self.cdf(X.flatten(), self.params)

    def score(self, X, y = None, cmp_fun = fpt.r2fun, n = 1001):
        # Compute the similarity between the cdf of the data and the cdf of the model.
        assert y is None, "y must be None"
        y = X.flatten() # The data 
        return fpt.compare_cdfs(self.predict, y, cmp_fun = cmp_fun, n = n)

    
    

    
