from matplotlib.pyplot import *
from numpy import *
import numpy as np
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma
from scipy.special import gamma as gamma_fun
from scipy.optimize import root_scalar, minimize_scalar
import fisher_plumes_tools as fpt

from collections import namedtuple

ExpParams   = namedtuple("ExpParams", "λ μ σ γ")
Exp2Params  = namedtuple("Exp2Params", "λ μ σp σn γ")
GammaParams = namedtuple("GammaParams", "λ μ σ γ k m")
params_str  = lambda params: " ".join([f"{p}={v:<10.2g}" for p,v in params._asdict().items()])

rand = np.random.rand
randn= np.random.randn

from scipy.special import gamma as gamma_fun
def agamma_cdf(params, y):
    λ = params.λ
    μ = params.μ
    k = params.k
    m = params.m

    Z0 = gamma_fun(m) * (μ**m)
    Z1 = gamma_fun(k) * (λ**k)
    Z  =  Z0 + Z1
    print(Z0, Z1, Z)    
    cdf = 0*y
    cdf[y<0]  = (1 - gamma_dist.cdf(abs(y[y<0]), m, scale = μ)) * Z0/Z
    cdf[y>=0] = Z0/Z + gamma_dist.cdf(y[y>=0], k, scale = λ) * Z1/Z
    return cdf

def cdf_gamma(params, y):
    λ, μ, k, m, σ, γ = [params._asdict()[k] for k in ['λ', 'μ', 'k', 'm', 'σ', 'γ']]    
    # Compute the CDF
    R = agamma_cdf(params, y)
    H = norm.cdf(y, scale = σ)
    return γ*R + (1-γ)*H

# Generate data from the model
def gen_data(M, exp_params, do_plot = False):
    # Generate the data
    λ, μ, σ, γ = exp_params
    y = zeros(M)
    z = rand(M) < γ
    i0= ~z
    n0= np.sum(~z)
    i1= z
    n1= np.sum(z)
    pp= λ/(λ + μ)
    ip= rand(M)<pp
    # Set y at ip to be sampled from an exponential with mean λ
    y[ip] = np.random.exponential(λ/2, sum(ip))
    # Set y at ~ip to be sampled from an exponential with mean μ
    y[~ip]= -np.random.exponential(μ/2, sum(~ip))       
    y[~z] = randn(n0)*σ
    
    labs = 0*z
    labs[z & (y>=0)] = 1
    labs[z & (y<0)]  = -1
    return y, labs

# Generate data from the model
def gen_gamma_data(M, gamma_params, do_plot = False):
    # Generate the data
    λ, μ, σ, γ, k, m = [gamma_params._asdict()[k] for k in ['λ', 'μ', 'σ', 'γ', 'k', 'm']]
    y = zeros(M)
    z = rand(M) < γ
    i0= ~z
    n0= np.sum(~z)
    i1= z
    n1= np.sum(z)
    pp= λ**k * gamma_fun(k)/(λ**k * gamma_fun(k) + μ**m * gamma_fun(m))
    print(f"{pp=:.2f}")
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


def plot_data(params, y, labs, figsize=None):
    if figsize is not None:
        figure(figsize=figsize)
    plot(y,color="gray")
    plot(where(labs>0)[0],  y[labs>0],  'b.', label="pos")
    plot(where(labs<0)[0],  y[labs<0],  'r.', label="neg")
    plot(where(labs==0)[0], y[labs==0], 'y.', label="noise")
    legend()
    title(params_str(params))

def print_params(p):
    print(params_str(p))
       
def plot_confusion_matrix(true_labs, pred_labs):
    cm = confusion_matrix(true_labs, pred_labs, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels = ["-1","0","1"]).plot()    

def cdf_data(y, n = 1001):
    ys = array(sorted(y))
    xv = linspace(ys[0], ys[-1], n)
    cdf = np.array([mean(ys<=x) for x in xv])
    return cdf, xv
    
def cdf_exp(exp_params, y):
    λ, μ, σ, γ = exp_params
    # Compute the CDF
    R = fpt.alaplace_cdf(λ, μ, y)
    H = norm.cdf(y, scale = σ)
    return γ*R + (1-γ)*H
    
def cdf_exp2(exp2_params, y):
    λ, μ, σp, σn, γ = exp2_params
    # Compute the CDF
    R = fpt.alaplace_cdf(λ, μ, y)
    Hn = norm.cdf(y, scale = σn)
    Hp = norm.cdf(y, scale = σp)
    H = Hp*(y>=0) + Hn*(y<0)
    return γ*R + (1-γ)*H

# def cdf_gamma(params, y):
#     λ, μ, σ, γ, k, m = params
#     # Compute the CDF
#     R = fpt.alaplace_cdf(λ, μ, y)
#     Hn = gamma_dist.cdf(y, m, scale = 1/μ)
#     Hp = gamma_dist.cdf(y, k, scale = 1/λ)
#     H = Hp*(y>=0) + Hn*(y<0)
#     return γ*R + (1-γ)*H

def run_em(y, β = 1, γ_init = 0.1, max_iter = 10, tol = 1e-6, beta_mean = 0, beta_strength = 0):
    # Initialize
    M     = len(y)
    γ_old = γ_init
    λ_old, μ_old, σ2_old = np.inf, np.inf, np.inf

    # Take the top M γ active values as those that active
    ind_z = argsort(-abs(y))[:int(M * γ_old)]    
    z = 0*y
    z[ind_z] = 1
    z = z.astype(bool)
    for i in range(max_iter):
        # E-step
        if i > 0:
            lrho = 0 * y
            lrho[y>=0] = log(2/(λ + μ))-2*abs(y[y>=0])/λ
            lrho[y<0]  = log(2/(λ + μ))-2*abs(y[y< 0])/μ
            leta       = -y**2/2/σ2 - log(sqrt(2*pi*σ2))
            z          = (lrho - leta) > log((1-min(γ,1-1e-8))/γ)
        # M-step
        γ = (sum(z) + beta_mean*beta_strength*len(z))/(len(z) + beta_strength*len(z))

        i0= ~z
        n0= sum(i0)        
        if n0>0:
            y0 = y[i0]
            σ2 = n0/(2*β) * (-1 + sqrt(1 + 4 * β * mean(y0**2)/n0))
        else:
            σ2 = var(y)*1e-3 # Don't make it exactly 0
                    
        ip  = z & (y>=0)
        i_n = z & (y<0)    
        n_p = sum(ip)
        nn  = sum(i_n)
        n   = n_p + nn
        ypm = mean(y[ip]) if n_p else 0
        ynm = mean(abs(y[i_n])) if nn else 0
        λ   = 2/n * (n_p*ypm + sqrt(n_p * nn * ypm * ynm))
        μ   = max(2/n * (nn*ynm + sqrt(n_p * nn * ypm * ynm)),1e-6)        

        # Print the values at the current iteration, including the iteration number
        print(f"Iter {i:>4d}: n+={sum(ip):>4d}, n-={sum(i_n):>4d} λ={λ:>6.2g}, μ={μ:>6.2g}, σ={sqrt(σ2):>6.2g}, γ={γ:>6.2g}")        
        # Check convergence
        if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ2 - σ2_old) < tol and abs(γ - γ_old) < tol:
            print("Converged.")
            break
        λ_old = λ
        μ_old = μ
        σ2_old = σ2
        γ_old = γ

    labs = 0*z
    labs[z & (y>=0)] = 1
    labs[z & (y<0)]  = -1
    return λ, μ, sqrt(σ2), γ, labs

def run_em2(y, β = 1, γ_init = 0.1, max_iter = 10, tol = 1e-6):
    # Initialize
    M     = len(y)
    γ_old = γ_init
    λ_old, μ_old, σ2p_old, σ2n_old = np.inf, np.inf, np.inf, np.inf

    # Take the top M γ active values as those that active
    ind_z = argsort(-abs(y))[:int(M * γ_old)]    
    z = 0*y
    z[ind_z] = 1
    z = z.astype(bool)
    for i in range(max_iter):
        # E-step
        if i > 0:
            lrho = 0 * y
            lrho[y>=0] = log(2/(λ + μ))-abs(y[y>=0])/λ
            lrho[y<0]  = log(2/(λ + μ))-abs(y[y< 0])/μ
            σ2 = 0*y
            σ2[y>=0] = σ2p
            σ2[y<0]  = σ2n
            leta     = -y**2/2/σ2 - log(sqrt(2*pi*σ2))
            z   = (lrho - leta) > log((1-min(γ,1-1e-8))/γ)
        # M-step
        γ = mean(z)

        i0p= ~z & (y>=0)
        n0p= sum(i0p)        
        if n0p>0:
            y0p = y[i0p]
            σ2p = n0p/(2*β) * (-1 + sqrt(1 + 4 * β * mean(y0p**2)/n0p))
        else:
            σ2p = var(y)*1e-3 # Don't make it exactly 0

        i0n= ~z & (y<0)
        n0n= sum(i0n)        
        if n0n>0:
            y0n = y[i0n]
            σ2n = n0n/(2*β) * (-1 + sqrt(1 + 4 * β * mean(y0n**2)/n0n))
        else:
            σ2n = var(y)*1e-3 # Don't make it exactly 0
                    
        ip = z & (y>=0)
        i_n= z & (y<0)
        n_p= sum(ip)
        nn = sum(i_n)
        n  = n_p + nn
        ypm= mean(y[ip]) if n_p else 0
        ynm= mean(abs(y[i_n])) if nn else 0
        λ = 2/n * (n_p*ypm + sqrt(n_p * nn * ypm * ynm))
        μ = max(2/n * (nn*ynm + sqrt(n_p * nn * ypm * ynm)),1e-6)
        

        # Print the values at the current iteration, including the iteration number
        print(f"Iter {i:>4d}: n+={sum(ip):>4d}, n-={sum(i_n):>4d} λ={λ:>6.2g}, μ={μ:>6.2g}, σp={sqrt(σ2p):>6.2g}, σn={sqrt(σ2n):>6.2g}, γ={γ:>6.2g}")                
        # Check convergence
        if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ2p - σ2p_old) < tol and abs(σ2n - σ2n_old) < tol  and abs(γ - γ_old) < tol:
            print("Converged.")
            break
        λ_old = λ
        μ_old = μ
        σ2p_old = σ2p
        σ2n_old = σ2n
        γ_old = γ

    labs = 0*z
    labs[z & (y>=0)] = 1
    labs[z & (y<0)]  = -1
    return λ, μ, sqrt(σ2p), sqrt(σ2n), γ, labs


def iter_fun(x0, f, max_iter = 1000, tol = 1e-6, damping = 0.5, verbose = False):
    x = x0
    for i in range(max_iter):
        x_new = f(x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = (1 - damping)*x_new + damping*x
        verbose and print(x)
        if any(np.isnan(x)):
            break

    if any(np.isnan(x)):
        print("Nans encountered.")
    elif i == max_iter - 1:
        print("Not converged.")
    else:
        verbose and print(f"Converged to {x} in {i:>4d} iterations.")
        
    return x_new


def run_gamma_em(y, params_init,
                 β = 1, max_iter = 10, tol = 1e-6, beta_mean = 0, beta_strength = 0,
                 ga_max = 1
                 ):
    # Initialize
    M     = len(y)

    λ_init = params_init.λ
    μ_init = params_init.μ
    σ_init = params_init.σ
    γ_init = params_init.γ
    k_init = 1
    m_init = 1
    
    σ2 = σ_init**2
    γ  = γ_init    
    λ  = λ_init
    μ  = μ_init
    k  = k_init
    m  = m_init

    λ_old, μ_old, σ2_old, γ_old, k_old, m_old = λ_init, μ_init, σ_init**2, γ_init, k_init, m_init

    ZFUN = lambda λ, μ, k, m: gamma(m) * μ**m + gamma(k) * λ**k
    for i in range(max_iter):
        Z = ZFUN(λ, μ, k, m)

        # E-step        
        lrho       = 0 * y
        lrho[y>=0] = -log(Z) + (k - 1)*log(y[y>=0]) - y[y>=0]/λ
        lrho[y<0]  = -log(Z) + (m - 1)*log(-y[y<0]) - abs(y[y<0])/μ
        leta       = -(y**2)/2/σ2 - log(sqrt(2*pi*σ2))
        z          = (lrho - leta) > log((1-min(γ,1-1e-8))/γ)

        # M-step        
        γ = (sum(z) + beta_mean*beta_strength*len(z))/(len(z) + beta_strength*len(z))

        i0= ~z
        n0= sum(i0)        
        if n0>0:
            y0 = y[i0]
            σ2 = n0/(2*β) * (-1 + sqrt(1 + 4 * β * mean(y0**2)/n0))
        else:
            σ2 = var(y)*1e-3 # Don't make it exactly 0
                    
        ip   = z & (y>=0)
        i_n  = z & (y<0)    
        n_p  = sum(ip)
        nn   = sum(i_n)
        n    = n_p + nn
        yps  = sum(y[ip]) if n_p else 0
        lyps = sum(log(y[ip])) if n_p else 0
        yns  = sum(abs(y[i_n])) if nn else 0
        lyns = sum(log(abs(y[i_n]))) if nn else 0

        λ, k, μ, m = iter_fun(np.array([λ_old, k_old, μ_old, m_old]), lambda x: np.array(
            [
                (ZFUN(x[0], x[2], x[1], x[3])/n/gamma(x[1])/x[1]*yps)**(1/(x[1]+1)),
                (log(x[0]) + digamma(x[1]))/x[0]*yps/lyps,
                (ZFUN(x[0], x[2], x[1], x[3])/n/gamma(x[3])/x[3]*yns)**(1/(x[3]+1)),
                (log(x[2]) + digamma(x[3]))/x[2]*yns/lyns
                ]))
        
        # Print the values at the current iteration, including the iteration number
        params = GammaParams(λ=λ, μ=μ, σ=sqrt(σ2), γ=γ, k=k, m=m)
        print(f"Iter {i:>4d}: n+={n_p:>4d}, n-={nn:>4d} n0={len(ip) -nn - n_p:>4d} " + params_str(params))
        # Check convergence
        if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ2 - σ2_old) < tol and abs(γ - γ_old) < tol and abs(m - m_old) < tol and abs(k - k_old) < tol:
            print("Converged.")
            break
        λ_old = λ
        μ_old = μ
        σ2_old = σ2
        γ_old = γ
        m_old = m
        k_old = k

    labs = 0*z
    labs[z & (y>=0)] = 1
    labs[z & (y<0)]  = -1
    params = GammaParams(λ=λ, μ=μ, σ=sqrt(σ2), γ=γ, k=k, m=m)
    return params, labs
