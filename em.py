from matplotlib.pyplot import *
from numpy import *
import numpy as np
from scipy.stats import norm
from scipy.special import gamma, digamma
from scipy.optimize import root_scalar, minimize_scalar
import fisher_plumes_tools as ftp

# Generate data from the model
def gen_data(M, λ, μ, σ, γ, do_plot = False):
    # Generate the data
    y = zeros(M)
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
    
    labs = 0*z
    labs[z & (y>=0)] = 1
    labs[z & (y<0)]  = -1
    return y, labs

def plot_data(λ, μ, σp, σn, γ, y, labs):
    figure(figsize=(12,4))
    plot(y,color="gray")
    plot(where(labs>0)[0],  y[labs>0],  'b.', label="pos")
    plot(where(labs<0)[0],  y[labs<0],  'r.', label="neg")
    plot(where(labs==0)[0], y[labs==0], 'y.', label="noise")
    legend()
    title(f"λ={λ:1.2g}, μ={μ:1.2g}, σp={σp:1.2g}, σn={σn:1.2g}, γ={γ:1.2g}")

def print_params(λ, μ, σp, σn, γ):
    print(f"λ={λ:1.2g}, μ={μ:1.2g}, σp={σp:1.2g}, σn={σn:1.2g}, γ={γ:1.2g}")   
   
def plot_confusion_matrix(true_labs, pred_labs):
    cm = confusion_matrix(true_labs, pred_labs, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels = ["-1","0","1"]).plot()    



def cdf(λ, μ, σp, σn, γ, y):
    # Compute the CDF
    R = fpt.alaplace_cdf(λ, μ, y)
    Hn = norm.cdf(y, scale = σn)
    Hp = norm.cdf(y, scale = σp)
    H = Hp*(y>=0) + Hn*(y<0)
    return γ*R + (1-γ)*H
    

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


def run_gamma_em(y, λ_init, μ_init, σ2_init, γ_init,
                 β = 1, max_iter = 10, tol = 1e-6, beta_mean = 0, beta_strength = 0,
                 ga_max = 1
                 ):
    # Initialize
    M     = len(y)

    γ_old = γ_init
    λ_old, μ_old, σ2_old = λ_init, μ_init, σ2_init
    
    k0_old = 1
    k1_old = 1

    σ2 = σ2_init
    γ  = γ_init    
    λ = λ_init
    μ = μ_init
    k0 = 1
    k1 = 1

    for i in range(max_iter):
        Z = gamma(k0) * μ**k0 + gamma(k1) * λ**k1
        # E-step        
        lrho = 0 * y
        lrho[y>=0] = -log(Z) + (k1 - 1)*log(y[y>=0]) - y[y>=0]/λ
        lrho[y<0]  = -log(Z) + (k0 - 1)*log(-y[y<0]) - abs(y[y<0])/μ
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
        yps = sum(y[ip]) if n_p else 0
        yns = sum(abs(y[i_n])) if nn else 0
        λ   = ((Z * yps)/n/k1/gamma(k1))**(1/(k1+1))
        μ   = ((Z * yns)/n/k0/gamma(k0))**(1/(k0+1))

        #print(f"{λ_old=} {μ_old=} {yps/n=} {k1_old=}.")
        f    = lambda k: np.abs((λ_old ** k) * gamma(k) * (log(λ_old) + digamma(k))/(gamma(k0_old) * μ_old**k0_old + gamma(k) * λ_old**k) - yps/n)
        sol1 = minimize_scalar(f, method="bounded", bounds=[1e-6,ga_max])
        k1   = sol1.x
        
        f    = lambda k: np.abs((μ_old ** k) * gamma(k) * (log(μ_old) + digamma(k))/(gamma(k) * μ_old**k + gamma(k1_old) * λ_old**k1_old) - yns/n)
        sol0 = minimize_scalar(f, method="bounded", bounds=[1e-6,ga_max])
        k0   = sol0.x
        
        # Print the values at the current iteration, including the iteration number
        print(f"Iter {i:>4d}: n+={n_p:>4d}, n-={nn:>4d} λ={λ:>6.2g}, μ={μ:>6.2g}, σ={sqrt(σ2):>6.2g}, γ={γ:>6.2g} k0={k0:>6.2g} k1={k1:>6.2g}")        
        # Check convergence
        if abs(λ - λ_old) < tol and abs(μ - μ_old) < tol and abs(σ2 - σ2_old) < tol and abs(γ - γ_old) < tol and abs(k0 - k0_old) < tol and abs(k1 - k1_old) < tol:
            print("Converged.")
            break
        λ_old = λ
        μ_old = μ
        σ2_old = σ2
        γ_old = γ
        k0_old = k0
        k1_old = k1

    labs = 0*z
    labs[z & (y>=0)] = 1
    labs[z & (y<0)]  = -1
    return λ, μ, sqrt(σ2), γ, k0, k1, labs
