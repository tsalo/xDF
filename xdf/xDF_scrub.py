#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:31:32 2019

@author: sorooshafyouni
University of Oxford, 2019
"""
import numpy as np
from AC_Utils import *
from MatMan import *
import os, sys
import itertools
import scipy.signal as ss
from scipy.linalg import toeplitz

def xDF_scrub(ts, T,\
              method      = 'truncate',\
              methodparam = 'adaptive',\
              verbose     = True,\
              TV          = True,\
              copy        = True):
    
    if copy: 
        ts = ts.copy()
    
    if np.shape(ts)[1] != T:
        if verbose: print('Second dimension should be T; the matrix was transposed')
        ts = np.transpose(ts)
    
    N  = np.shape(ts)[0]

    scrubbed_frames = np.where(np.any(np.isnan(ts), axis=0))[0]
    t = T - len(scrubbed_frames)

    ts = ts - np.nanmean(ts, axis=1)[:, None]
    norms = np.nanmean(ts**2, axis=1)[:, None]

    # Mask NaNs with 0s for summation
    if t < T: ts[:, scrubbed_frames] = 0

    # Calculate denominator according to acf in R
    list_pairs = list(itertools.combinations(np.setdiff1d(np.arange(T), scrubbed_frames), 2))
    n_pairs = [t] + list(np.bincount([y - x for (x, y) in list_pairs], minlength=T)[1:])
    n_denom = n_pairs + np.arange(T)

    # Calculate autocorrelation
    ac = np.concatenate([ss.correlate(ts[i, :], ts[i, :], method='fft')[None, -T:] / n_denom for i in np.arange(N)], axis=0) / norms

    # Calculate cross-correlation
    xc = np.zeros((N, N, 2 * T - 1))
    XX = np.triu_indices(N, 1)[0]
    YY = np.triu_indices(N, 1)[1]

    for (i, j) in zip(XX, YY):
    
        xc[i, j, :] = ss.correlate(ts[i, :], ts[j, :], method='fft') / (np.sqrt(norms[i] * norms[j]) * np.concatenate((np.flip(np.delete(n_denom, 0)), n_denom)))
    
    xc = xc + np.transpose(xc, (1, 0, 2))

    # Extract positive and negative cross-correlations
    xc_p = xc[:, :, :(T-1)]
    xc_p = np.flip(xc_p, axis = 2)
    xc_n = xc[:, :, -(T-1):] 

    # Extract lag-0 correlations
    rho = np.eye(N) + xc[:, :, T-1]

    # Regularize!
    if method.lower() == 'tukey':
        if methodparam == '':
            M = np.sqrt(T)
        else: M = methodparam
        if verbose: print('AC regularization: Tukey tapering of M = ' + str(int(np.round(M))))
        ac   = tukeytaperme(ac, T-1, M)
        xc_p = tukeytaperme(xc_p, T-1, M)
        xc_n = tukeytaperme(xc_n, T-1, M)
        
    elif method.lower() == 'truncate':

        if type(methodparam) == str: # Adaptive truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError('What?! Choose adaptive as the option or pass an integer for truncation')
            if verbose: print('AC regularization: adaptive truncation')
           
            ac, bp = shrinkme(ac, T)
            
            for i in np.arange(N):
                for j in np.arange(N):

                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = curbtaperme(xc_p[i, j, :], T-1, maxBP, verbose=False)
                    xc_n[i, j, :] = curbtaperme(xc_n[i, j, :], T-1, maxBP, verbose=False)

        elif type(methodparam) == int: # Non-adaptive truncation
            if verbose: print('AC regularization: non-adaptive truncation on M = ' + str(methodparam))         
            ac    = curbtaperme(ac, T-1, methodparam)
            xc_p  = curbtaperme(xc_p, T-1, methodparam)
            xc_n  = curbtaperme(xc_n, T-1, methodparam)
            
        else: raise ValueError('Method parameter for truncation method should be either str or int')
    
    # Estimate variance (big formula)
    VarHatRho = np.zeros((N, N))

    for (i, j) in zip(XX, YY):
        
        r = rho[i, j]

        Sigx = toeplitz(ac[i, :])
        Sigy = toeplitz(ac[j, :])

        Sigx = np.delete(Sigx, scrubbed_frames, axis=0)
        Sigx = np.delete(Sigx, scrubbed_frames, axis=1)
        Sigy = np.delete(Sigy, scrubbed_frames, axis=0)
        Sigy = np.delete(Sigy, scrubbed_frames, axis=1)

        Sigxy = np.triu(toeplitz(np.insert(xc_p[i, j, :], 0, r)), k=1) + \
                    np.tril(toeplitz(np.insert(xc_n[i, j, :], 0, r)))
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=0)
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=1)
        Sigyx = np.transpose(Sigxy)

        VarHatRho[i, j] = ((r**2 / 2) * np.trace(Sigx @ Sigx) + (r**2 / 2) * np.trace(Sigy @ Sigy) \
                            + r**2 * np.trace(Sigyx @ Sigxy) + np.trace(Sigxy @ Sigxy) + np.trace(Sigx @ Sigy) \
                            - 2 * r * np.trace(Sigx @ Sigxy) - 2 * r * np.trace(Sigy @ Sigyx)) / t**2

    VarHatRho = VarHatRho + np.transpose(VarHatRho)
        
    # Truncate to theoretical variance
    if TV: VarHatRho = np.maximum(VarHatRho, (1 - rho**2)**2 / t)
    
    return VarHatRho

##############################################################################
########################## END OF xDF_Calc ###################################
##############################################################################
    
# Disable verbose
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore verbose
def enablePrint():
    sys.stdout = sys.__stdout__    