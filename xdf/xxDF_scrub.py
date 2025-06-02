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
import pandas as pd

def fast_trace(A, B): return(np.einsum('ij,ji->', A, B))

def xDF_2nd_scrub(ts, T,\
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
    
    N = np.shape(ts)[0]

    scrubbed_frames = np.where(np.any(np.isnan(ts), axis=0))[0]
    T_scrubbed = T - len(scrubbed_frames)

    ts = ts - np.nanmean(ts, axis=1)[:, None]
    norms = np.nanmean(ts**2, axis=1)[:, None]

    # Mask NaNs with 0s for summation
    if T_scrubbed < T: ts[:, scrubbed_frames] = 0

    # Calculate denominator according to acf in R
    list_pairs = list(itertools.combinations(np.setdiff1d(np.arange(T), scrubbed_frames), 2))
    n_pairs = [T_scrubbed] + list(np.bincount([y - x for (x, y) in list_pairs], minlength=T)[1:])
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
        ac = tukeytaperme(ac, T-1, M)
        xc_p = tukeytaperme(xc_p, T-1, M)
        xc_n = tukeytaperme(xc_n, T-1, M)
        
    elif method.lower() == 'truncate':

        if type(methodparam) == str: # Adaptive truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError('What?! Choose adaptive as the option or pass an integer for truncation')
            if verbose: print('AC regularization: adaptive truncation')
           
            ac, bp = shrinkme(ac, T)
            
            for (i, j) in itertools.product(np.arange(N), np.arange(N)):
    
                maxBP = np.max([bp[i], bp[j]])
                xc_p[i, j, :] = curbtaperme(xc_p[i, j, :], T-1, maxBP, verbose=False)
                xc_n[i, j, :] = curbtaperme(xc_n[i, j, :], T-1, maxBP, verbose=False)

        elif type(methodparam) == int: # Non-adaptive truncation
            if verbose: print('AC regularization: non-adaptive truncation on M = ' + str(methodparam))         
            ac = curbtaperme(ac, T-1, methodparam)
            xc_p = curbtaperme(xc_p, T-1, methodparam)
            xc_n = curbtaperme(xc_n, T-1, methodparam)
            
        else: raise ValueError('Method parameter for truncation method should be either str or int')
    
    # Estimate variance (big formula)
    xc_m = np.zeros((N, N, T_scrubbed, T_scrubbed))

    for (i, j) in zip(XX, YY):

        Sigxy = np.triu(toeplitz(np.insert(xc_p[i, j, :], 0, 0)), k=1) + np.tril(toeplitz(np.insert(xc_n[i, j, :], 0, rho[i, j])))
    
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=0)
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=1)
        Sigyx = np.transpose(Sigxy)
    
        xc_m[i, j, :, :] = Sigxy
        xc_m[j, i, :, :] = Sigyx

    for i in range(N):
    
        Sigx = toeplitz(ac[i, :])
        Sigx = np.delete(Sigx, scrubbed_frames, axis=0)
        Sigx = np.delete(Sigx, scrubbed_frames, axis=1)
    
        xc_m[i, i, :, :] = Sigx

    pairs = list(itertools.combinations(np.arange(N), 2))
    row_index = pd.MultiIndex.from_tuples(pairs, names=['R1', 'R2'])
    col_index = pd.MultiIndex.from_tuples(pairs, names=['C1', 'C2'])

    pw_vars = pd.DataFrame(np.full((len(pairs), len(pairs)), np.nan), index=row_index, columns=col_index)

    for ((s, t), (x, y)) in itertools.combinations_with_replacement(list(itertools.combinations(np.arange(N), 2)), 2):
    
        cov = np.array([[2 * np.linalg.norm(xc_m[x, s, :, :])**2, 2 * np.linalg.norm(xc_m[y, s, :, :])**2, 2 * fast_trace(xc_m[x, s, :, :], xc_m[s, y, :, :])],
                        [2 * np.linalg.norm(xc_m[x, t, :, :])**2, 2 * np.linalg.norm(xc_m[y, t, :, :])**2, 2 * fast_trace(xc_m[x, t, :, :], xc_m[t, y, :, :])],
                        [2 * fast_trace(xc_m[x, s, :, :], xc_m[t, x, :, :]), 2 * fast_trace(xc_m[y, s, :, :], xc_m[t, y, :, :]), fast_trace(xc_m[y, s, :, :], xc_m[t, x, :, :]) + fast_trace(xc_m[x, s, :, :], xc_m[t, y, :, :])]])
    
        v1 = np.array([[2 * np.linalg.norm(xc_m[s, s, :, :])**2, 2 * np.linalg.norm(xc_m[t, s, :, :])**2, 2 * fast_trace(xc_m[s, s, :, :], xc_m[s, t, :, :])],
                       [2 * np.linalg.norm(xc_m[s, t, :, :])**2, 2 * np.linalg.norm(xc_m[t, t, :, :])**2, 2 * fast_trace(xc_m[s, t, :, :], xc_m[t, t, :, :])],
                       [2 * fast_trace(xc_m[s, s, :, :], xc_m[t, s, :, :]), 2 * fast_trace(xc_m[t, s, :, :], xc_m[t, t, :, :]), fast_trace(xc_m[t, s, :, :], xc_m[t, s, :, :]) + fast_trace(xc_m[s, s, :, :], xc_m[t, t, :, :])]])
    
        v2 = np.array([[2 * np.linalg.norm(xc_m[x, x, :, :])**2, 2 * np.linalg.norm(xc_m[y, x, :, :])**2, 2 * fast_trace(xc_m[x, x, :, :], xc_m[x, y, :, :])],
                       [2 * np.linalg.norm(xc_m[x, y, :, :])**2, 2 * np.linalg.norm(xc_m[y, y, :, :])**2, 2 * fast_trace(xc_m[x, y, :, :], xc_m[y, y, :, :])],
                       [2 * fast_trace(xc_m[x, x, :, :], xc_m[y, x, :, :]), 2 * fast_trace(xc_m[y, x, :, :], xc_m[y, y, :, :]), fast_trace(xc_m[y, x, :, :], xc_m[y, x, :, :]) + fast_trace(xc_m[x, x, :, :], xc_m[y, y, :, :])]])
        
        h1 = 1 / np.array([np.trace(xc_m[s, s, :, :]), np.trace(xc_m[t, t, :, :]), np.trace(xc_m[s, t, :, :])])
        hess1 = rho[s, t] * np.outer(h1, h1) * np.array([[3/4, 1/4, -1/2], [1/4, 3/4, -1/2], [-1/2, -1/2, 0]])
        grad1 = rho[s, t] * h1 * np.array([-1/2, -1/2, 1])
        
        h2 = 1 / np.array([np.trace(xc_m[x, x, :, :]), np.trace(xc_m[y, y, :, :]), np.trace(xc_m[x, y, :, :])])
        hess2 = rho[x, y] * np.outer(h2, h2) * np.array([[3/4, 1/4, -1/2], [1/4, 3/4, -1/2], [-1/2, -1/2, 0]])
        grad2 = rho[x, y] * h2 * np.array([-1/2, -1/2, 1])
        
        pw_vars.loc[(s, t), (x, y)] = grad2 @ cov @ grad1 / 2 + grad1 @ cov @ grad2 / 2 - np.trace(hess1 @ v1) * np.trace(hess2 @ v2) / 4

    pw_vars = pw_vars.where(pw_vars.notna(), pw_vars.T.values)

    if TV: np.fill_diagonal(pw_vars.values, np.maximum(np.diagonal(pw_vars.values), (1 - np.array([rho[i, j] for (i, j) in pairs])**2)**2 / T_scrubbed))
    
    return pw_vars

##############################################################################
########################## END OF xDF_Calc ###################################
##############################################################################
    
# Disable verbose
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore verbose
def enablePrint():
    sys.stdout = sys.__stdout__    