import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def integ(xs,ys):
    Y = []
    X = []
    temp_y = 0.0
    temp_x = 0.0
    for i in range(len(xs)-1):
        temp_x = (xs[i+1]+xs[i])/2
        X.append(temp_x)

        temp_y += (xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2
        Y.append(temp_y)
    return np.array(X),np.array(Y)

def diff(x,y):
    h = x[1]-x[0]
    y0 = (-3*y[0]+4*y[1]-y[2])/(2*h)
    y1 = (y[2:]-y[:-2])/(2*h)
    y2 = (y[-3]-4*y[-2]+3*y[-1])/(2*h)
    return np.concatenate(([y0],y1,[y2]))

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

"""def hankel_matrix(array, l):

    HM = np.zeros([series.shape[0] - l, l])

    for i in range(0, series.shape[0] - l):
        HM[i] = np.squeeze(series[i:i + l, 0])
    return HM"
"""

def hankel_matrix(array, l, each = 1):
    array = array[::each]
    HM = np.zeros([len(array) - l, l])

    for i in range(0, len(array) - l):
        HM[i] = np.squeeze(array[i:i + l])
    return HM

def eigenvalues_plot(__x,n,each = 1,num_l = None, MinMax = False):
    HM = hankel_matrix(__x, n,each)

    S = np.linalg.svd(HM,compute_uv=False)
    if MinMax:
        S /= max(S)

    plt.figure(figsize=(11, 6))

    plt.plot(S[:num_l]**.5,'o')
    plt.plot(S[:num_l]**.5,'--')

    #plt.xlim(-1,20)


    plt.xlabel('N', size=23)
    plt.ylabel('λ', size=23)

    plt.tick_params(axis='both', which='major', labelsize=25,length=8, width=4)
    plt.grid(which='major',
        color = 'gray',
        linewidth = 0.8)
    plt.minorticks_on()
    plt.grid(which='minor',
        color = 'gray',
        linestyle = '--',
        linewidth = 0.5)
    plt.legend(shadow=True, ncol=1, fontsize=15)
    plt.show()

class NWregression():

    def __init__(self, h, kernel = 'gaussian', metric = 'l2'):
        kernels = {
            'gaussian': self._gaussian_kernel,
            'rectangular': self._rectangular_kernel,
            'triangular': self._triangular_kernel,
            'quadratic': self._quadratic_kernel,
            'quartic': self._quartic_kernel
        }
        self.metric = metric
        self.h = h
        self.kernel = kernels[kernel]
        
    def _gaussian_kernel(self, r):
        return np.exp(-2*r**2)
    
    def _rectangular_kernel(self, r):
        return r * (r < 1)

    def _triangular_kernel(self, r):
        return (1-r) * (r < 1)

    def _quadratic_kernel(self, r):
        return (1-r**2) * (r < 1)

    def _quartic_kernel(self, r):
        return (1-r**2)**2 * (r < 1)
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        return self

    def predict(self, X):
        
        dist_matrix = pairwise_distances(X,self.X, metric = self.metric)
        
        weight = self.kernel(dist_matrix/self.h)
        
        norm_coef = np.sum(weight,axis=1).reshape((len(X),1))
        
        regression_ans = (weight@self.Y)/norm_coef
        
        return regression_ans
