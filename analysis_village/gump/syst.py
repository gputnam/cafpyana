import numpy as np

class SystematicList(object):
    def __init__(self, systs):
        self.systs = systs

    def cov(self, var, cut, bins, NCV, shapeonly=False):
        if len(self.systs) == 0:
            return np.zeros((NCV.size, NCV.size))
            
        return np.sum([s.cov(var, cut, bins, NCV, shapeonly=shapeonly) for s in self.systs], axis=0)

class Systematic(object):
    def __init__(self):
        pass
        
    def nuniv(self):
        pass
        
    def univ(self, var, cut, bins, i_univ):
        pass

    # Whether to average the separate universes, or not (i.e. treat them as different uncertainties)
    def avg(self):
        return True # true by default
    
    def cov(self, var, cut, bins, NCV, shapeonly=False):
        if shapeonly:
            diff = (bins[1:] - bins[:-1])
            norm = np.sum(NCV*diff)
            if norm > 1e-5:
                NCV = NCV / norm
        
        N_univ = []
        for i_univ in range(self.nuniv()):
            N = self.univ(var, cut, bins, i_univ)
            if shapeonly:
                diff = (bins[1:] - bins[:-1])
                norm = np.sum(N*diff)
                if norm > 1e-5:
                    N = N / norm
                
            N_univ.append(N)
    
        cov =  np.sum([np.outer(N - NCV, N - NCV) for N in N_univ], axis=0)
        if self.avg():
            cov = cov / self.nuniv()

        return cov

class SystSampleSystematic(Systematic):
    def __init__(self, df, scale="glob_scale", norm=1.):
        self.df = df
        self.scale = scale
        self.norm = norm
        
    def nuniv(self):
        return 1
        
    def cov(self, var, cut, bins, NCV, shapeonly=False):
        self.CV = NCV
        return super().cov(var, cut, bins, NCV, shapeonly=shapeonly)

    def univ(self, var, cut, bins, i_univ):
        assert(i_univ == 0)
        if not isinstance(var, list):
            var = [var]
            bins = [bins]

        return np.histogramdd([self.df.loc[self.df[cut], v] for v in var], bins=bins, weights=self.df.loc[self.df[cut], self.scale])[0].flatten()*self.norm + self.CV

class StatSampleSystematic(object):
    def __init__(self, df, scale="glob_scale", norm=1):
        self.df = df
        self.scale = scale
        self.norm = norm
        
    def cov(self, var, cut, bins, NCV, shapeonly=False):
        if not isinstance(var, list):
            var = [var]
            bins = [bins]

        # Poisson variance of weighted events is square of weights
        w = self.df.loc[self.df[cut], self.scale]**2
        var = np.histogramdd([self.df.loc[self.df[cut], v] for v in var], bins=bins, weights=w)[0].flatten()*self.norm
        return np.diag(var)

class CorrelatedSystematic(Systematic):
    def __init__(self, a, b):        
        self.systa = a
        self.systb = b

    def nuniv(self):
        return self.systa.nuniv()

    def univ(self, var, cut, bins, i_univ):
        Na = self.systa.univ(var, cut, bins, i_univ)
        Nb = self.systb.univ(var, cut, bins, i_univ)
        N = np.concatenate((Na, Nb))
        return N

class UnCorrelatedSystematic(object):
    def __init__(self, a, b):
        self.systa = a
        self.systb = b

    def cov(self, var, cut, bins, NCV, shapeonly=False):
        NCVa = NCV[:NCV.size//2]
        NCVb = NCV[NCV.size//2:]
        cova = self.systa.cov(var, cut, bins, NCVa, shapeonly=shapeonly)
        covb = self.systb.cov(var, cut, bins, NCVb, shapeonly=shapeonly)
        cov = np.zeros((cova.shape[0]*2, cova.shape[1]*2))
        cov[:cova.shape[0], :cova.shape[1]] = cova[:]
        cov[cova.shape[0]:, cova.shape[1]:] = covb[:]
        return cov
        
class SampleSystematic(Systematic):
    def __init__(self, df, scale="glob_scale"):
        self.df = df
        self.scale = scale
        
    def nuniv(self):
        return 1
        
    def univ(self, var, cut, bins, i_univ):
        assert(i_univ == 0)

        if not isinstance(var, list):
            var = [var]
            bins = [bins]

        return np.histogramdd([self.df.loc[self.df[cut], v] for v in var], bins=bins, weights=self.df.loc[self.df[cut], self.scale])[0].flatten()

class WeightSystematic(Systematic):
    def __init__(self, df, wgts, avg=True, scale="glob_scale"):
        self.df = df
        self.wgts = wgts
        self._nuniv = len(wgts)
        self.scale = scale
        self._avg = avg
        
    def nuniv(self):
        return self._nuniv

    def avg(self):
        return self._avg
        
    def univ(self, var, cut, bins, i_univ):
        if not isinstance(var, list):
            var = [var]
            bins = [bins]

        wgt_v = self.df[self.scale] * self.df[self.wgts[i_univ]]
        return np.histogramdd([self.df.loc[self.df[cut], v] for v in var], bins=bins, weights=wgt_v[self.df[cut]])[0].flatten()

