import numpy as np
import yaml
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer
from collections import namedtuple
import logging, utils
from scipy.stats import wilcoxon

logger = utils.create_logger("assumptions")
logger.setLevel(logging.DEBUG)
INFO  = logger.info
WARN  = logger.warning
DEBUG = logger.debug

# Compute the Szekely's energy test 
class Energy:
    @staticmethod
    def stat(X, Y):
        n1 = X.shape[0]
        n2 = Y.shape[0]
        n = n1 + n2
        XY = np.vstack([X,Y])
        D = squareform(pdist(XY))
        DXX = D[:n1, :n1]
        DYY = D[n1:, n1:]
        DXY = D[:n1, n1:]
        E = 2 * np.mean(DXY) - np.mean(DXX) - np.mean(DYY)
        return E * (n1 * n2) 

    @staticmethod
    def test(X, Y, n=100):
        n1, n2 = X.shape[0], Y.shape[0]
        XY = np.vstack([X,Y])
        Eobs = Energy.stat(X, Y)
        Eperm = np.zeros(n)
        for i in range(n):
            np.random.shuffle(XY)
            Eperm[i] = Energy.stat(XY[:n1], XY[n1:])
        pval = np.mean(Eperm > Eobs)
        return pval

    @staticmethod
    def test_gaussian(X, n_rand = None, **kwargs):
        if n_rand is None:
            n_rand = len(X)
    
        Xm = np.mean(X, axis=0)
        Xcov = np.cov(X.T)
        Y = np.random.multivariate_normal(Xm, Xcov, n_rand)
        return Energy.test(X,Y,**kwargs)
    
def squareform(vec, incl_diag=True):
    # Len(vec) = n(n+1)/2
    if not incl_diag:
        n = int(np.sqrt(0.25 + 2*len(vec)) + 0.5)
    else:
        n = int(np.sqrt(0.25 + 2*len(vec)) - 0.5)
    M = np.zeros((n,n))
    M[np.triu_indices(n,1 - incl_diag)] = vec
    M += M.T
    if incl_diag:
        M[np.diag_indices(n)] /= 2
    return M.astype(type(vec[0]))

# Need these outside the class to avoid pickling issues
LocIndKey    = namedtuple("LocIndKey", ["i1", "i2", "src1", "src2", "ifreq"]) 
LocIndResult = namedtuple("LocIndResult", ["pval"])
class LocationIndependence:
    @staticmethod
    def results_to_vec(results):
        keys = sorted(list(results.keys()),key=lambda x: x.ifreq*10**8 + x.i2 + x.i1*10**4)
        ifreqs = {k.ifreq for k in keys}
        vec  = np.array([results[k].pval for k in keys])
        assert len(vec) % len(ifreqs) == 0, f"Number of results ({len(vec)}) not a multiple of number of frequencies ({len(ifreqs)})."
        stride = len(vec) // len(ifreqs)
        keys=[keys[i*stride:(i+1)*stride] for i in range(len(ifreqs))]
        ifreqs = [k[0].ifreq for k in keys]
        return vec.reshape((len(ifreqs), -1)), ifreqs, keys
    @staticmethod
    def run(fp_data, assm_spec, ifreqs):
        spec = assm_spec["location_independence"]
        DEBUG(f"Testing location independence for {spec=} and {ifreqs=}")
        iprb = spec["iprb"]

        ss = fp_data["ss"]
        cc = fp_data["cc"]

        assert (iprb < len(ss)) and (iprb < len(cc)), f"Invalid probe index: {iprb}"

        ss, cc = ss[iprb], cc[iprb]
        
        srcs = sorted(list(ss.keys()))
        assert len(srcs)>1, f"Need at least 2 sources to test location independence, found {len(srcs)}."

        
        n_freqs = ss[srcs[0]].shape[-1]
        if ifreqs is None:
            ifreqs = list(range(n_freqs))
        else:
            ifreqs = [i[0] for i in ifreqs]
            assert all([0 <= i < n_freqs for i in ifreqs]), f"Invalid frequency indices: {ifreqs}"
        
        DEBUG(f"{len(srcs)} sources and {len(ifreqs)} frequencies.")
        DEBUG(f"{ifreqs=}")

        results = {}
        np.random.seed(spec["seed"])
        for i1, s1 in enumerate(srcs):
            for i2 in range(i1, len(srcs)):
                s2 = srcs[i2]
                for ii, ifreq in enumerate(ifreqs):
                    a = cc[s1][0,:,ifreq]
                    b = ss[s1][0,:,ifreq]
                    c = cc[s2][0,:,ifreq]
                    d = ss[s2][0,:,ifreq]
                    X = np.array([a,b]).T
                    Y = np.array([c,d]).T
                    key    = LocIndKey(i1=i1, i2=i2, src1=s1, src2=s2, ifreq=ifreq)
                    estat  = Energy.test(X,Y,spec["n_perm"])
                    results[key] = LocIndResult(pval=estat)
                    DEBUG(f'{key=}: {estat=}')
                    
        return results

# Need these outside the class to avoid pickling issues
GaussCoefsKey    = namedtuple("GaussCoefsKey",    ["i1", "src1", "ifreq"]) 
GaussCoefsResult = namedtuple("GaussCoefsResult", ["pvals"])
class GaussianCoefs:
    @staticmethod
    def results_to_vec(results, summary = np.mean):
        keys = sorted(list(results.keys()),key=lambda x: x.ifreq*10**8 + x.i1*10**4)
        ifreqs = {k.ifreq for k in keys}
        vec  = np.array([summary(results[k].pvals) for k in keys])
        assert len(vec) % len(ifreqs) == 0, f"Number of results ({len(vec)}) not a multiple of number of frequencies ({len(ifreqs)})."
        stride = len(vec) // len(ifreqs)
        keys=[keys[i*stride:(i+1)*stride] for i in range(len(ifreqs))]
        ifreqs = [k[0].ifreq for k in keys]
        return vec.reshape((len(ifreqs), -1)), ifreqs, keys
    @staticmethod
    def run(fp_data, assm_spec, ifreqs):
        spec = assm_spec["gaussian_coefs"]
        iprb = spec["iprb"]
        n_trials = spec["n_trials"]
        DEBUG(f"Testing gaussian coefficients for {spec=} and {ifreqs=} for probe {iprb} and {n_trials} trials.")

        ss = fp_data["ss"]
        cc = fp_data["cc"]

        assert (iprb < len(ss)) and (iprb < len(cc)), f"Invalid probe index: {iprb}"

        ss, cc = ss[iprb], cc[iprb]
        
        srcs = sorted(list(ss.keys()))
        
        n_freqs = ss[srcs[0]].shape[-1]
        if ifreqs is None:
            ifreqs = list(range(n_freqs))
        else:
            ifreqs = [i[0] for i in ifreqs]
            assert all([0 <= i < n_freqs for i in ifreqs]), f"Invalid frequency indices: {ifreqs}"
        
        DEBUG(f"{len(srcs)} sources and {len(ifreqs)} frequencies.")
        DEBUG(f"{ifreqs=}")

        results = {}
        np.random.seed(spec["seed"])
        for i1, s1 in enumerate(srcs):
            for ii, ifreq in enumerate(ifreqs):
                a = cc[s1][0,:,ifreq]
                b = ss[s1][0,:,ifreq]
                X = np.array([a,b]).T
                key    = GaussCoefsKey(i1=i1, src1=s1, ifreq=ifreq)
                pvals  = []
                for trial in range(n_trials):
                    pvals.append(Energy.test_gaussian(X,spec["n_perm"]))
#                    pvals.append(0)
                    DEBUG(f'{key=},{trial=}: {pvals[-1]}')                    
                results[key] = GaussCoefsResult(pvals=pvals) 
                    
        return results

# Need these outside the class to avoid pickling issues
StationarityKey    = namedtuple("StationarityKey",    ["i1", "src1", "ifreq"]) 
StationarityResult = namedtuple("StationarityResult", ["same_dist", "sin_mean_0","cos_mean_0","sin_cos_corr"])
class Stationarity:
    @staticmethod
    def results_to_vec(results, summary = np.mean):
        keys = sorted(list(results.keys()),key=lambda x: x.ifreq*10**8 + x.i1*10**4)
        ifreqs = {k.ifreq for k in keys}
        vec  = np.array([summary(results[k].pvals) for k in keys])
        assert len(vec) % len(ifreqs) == 0, f"Number of results ({len(vec)}) not a multiple of number of frequencies ({len(ifreqs)})."
        stride = len(vec) // len(ifreqs)
        keys=[keys[i*stride:(i+1)*stride] for i in range(len(ifreqs))]
        ifreqs = [k[0].ifreq for k in keys]
        return vec.reshape((len(ifreqs), -1)), ifreqs, keys
    @staticmethod
    def run(fp_data, assm_spec, ifreqs):
        spec = assm_spec["stationarity"]
        iprb = spec["iprb"]
        DEBUG(f"Testing stationarity for {spec=} and {ifreqs=} for probe {iprb} and {n_trials} trials.")

        ss = fp_data["ss"]
        cc = fp_data["cc"]

        assert (iprb < len(ss)) and (iprb < len(cc)), f"Invalid probe index: {iprb}"

        ss, cc = ss[iprb], cc[iprb]
        
        srcs = sorted(list(ss.keys()))
        
        n_freqs = ss[srcs[0]].shape[-1]
        if ifreqs is None:
            ifreqs = list(range(n_freqs))
        else:
            ifreqs = [i[0] for i in ifreqs]
            assert all([0 <= i < n_freqs for i in ifreqs]), f"Invalid frequency indices: {ifreqs}"
        
        DEBUG(f"{len(srcs)} sources and {len(ifreqs)} frequencies.")
        DEBUG(f"{ifreqs=}")

        results = {}
        np.random.seed(spec["seed"])
        for i1, s1 in enumerate(srcs):
            for ii, ifreq in enumerate(ifreqs):
                a = cc[s1][0,:,ifreq]
                b = ss[s1][0,:,ifreq]
                X = np.array([a,b]).T
                key = StationarityKey(i1=i1, src1=s1, ifreq=ifreq)
                
                results[key] = StationarityResult(
                    same_dist    = Energy.test(a,b,spec["n_perm"]),
                    cos_mean_0   = wilcoxon(a).pvalue,
                    sin_min_0    = wilcoxon(b).pvalue,
                    sin_cos_corr = np.dot(a,b),
                )

                DEBUG(f'{key=}: {results[key]}')
                    
        return results
    
class TestAssumptions:
    valid_tests = {"location_independence":LocationIndependence,
                   "gaussian_coefs":GaussianCoefs,
                   "stationarity":Stationarity,
                   }
    def __init__(self, assm_yaml, fp_data):
        self.assm_spec = yaml.load(open(assm_yaml, 'r'), Loader=yaml.FullLoader)
        self.fp_data = fp_data
        
    def run(self, ifreqs = None):
        results = {}
        for fld in self.assm_spec:
            if fld in TestAssumptions.valid_tests:
                if not self.assm_spec[fld]["run"]:
                    DEBUG(f"Skipping {fld} test.")
                    continue

                results[fld] = TestAssumptions.valid_tests[fld].run(self.fp_data, self.assm_spec, ifreqs)

        return results
        
