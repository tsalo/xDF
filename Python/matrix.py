#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matrix operations for xDF.

Created on Fri Jan 11 16:41:29 2019

Many of these functions were copy pasted from bctpy package:
    https://github.com/aestrivex/bctpy
under GNU V3.0:
    https://github.com/aestrivex/bctpy/blob/master/LICENSE

@author: sorooshafyouni
University of Oxford, 2019
"""
import numpy as np
import statsmodels.stats.multitest as smmt
from scipy import stats


def SumMat(Y0, T, copy=True):
    """Perform element-wise summation of each row with other rows.

    Parameters
    ----------
    Y0 : a 2D matrix of size TxN

    Returns
    -------
    SM : 3D matrix, obtained from element-wise summation of each row with other
         rows.

    SA, Ox, 2019
    """

    if copy:
        Y0 = Y0.copy()

    if np.shape(Y0)[0] != T:
        print("SumMat::: Input should be in TxN form, the matrix was transposed.")
        Y0 = np.transpose(Y0)

    N = np.shape(Y0)[1]
    Idx = np.triu_indices(N)
    # F = (N*(N-1))/2
    SM = np.empty([N, N, T])
    for i in np.arange(0, np.size(Idx[0]) - 1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = Y0[:, xx] + Y0[:, yy]
        SM[yy, xx, :] = Y0[:, yy] + Y0[:, xx]

    return SM


def ProdMat(Y0, T, copy=True):
    """Perform element-wise multiplication of each row with other rows.

    Parameters
    ----------
    Y0 : a 2D matrix of size TxN

    Returns
    -------
    SM : 3D matrix, obtained from element-wise multiplication of each row with
         other rows.

    SA, Ox, 2019
    """

    if copy:
        Y0 = Y0.copy()

    if np.shape(Y0)[0] != T:
        print("ProdMat::: Input should be in TxN form, the matrix was transposed.")
        Y0 = np.transpose(Y0)

    N = np.shape(Y0)[1]
    Idx = np.triu_indices(N)
    # F = (N*(N-1))/2
    SM = np.empty([N, N, T])
    for i in np.arange(0, np.size(Idx[0]) - 1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = Y0[:, xx] * Y0[:, yy]
        SM[yy, xx, :] = Y0[:, yy] * Y0[:, xx]

    return SM


def CorrMat(ts, T, method="rho", copy=True):
    """Produce sample correlation matrices or naively corrected z maps."""

    if copy:
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        print("xDF::: Input should be in IxT form, the matrix was transposed.")
        ts = np.transpose(ts)

    N = np.shape(ts)[0]
    R = np.corrcoef(ts)

    Z = np.arctanh(R) * np.sqrt(T - 3)

    R[range(N), range(N)] = 0
    Z[range(N), range(N)] = 0

    return R, Z


def stat_threshold(Z, mce="fdr_bh", a_level=0.05, side="two", copy=True):
    """Threshold z maps.

    Parameters
    ----------

    mce: multiple comparison error correction method, should be
    among of the options below. [defualt: 'fdr_bh'].
    The options are from statsmodels packages:

        `b`, `bonferroni` : one-step correction
        `s`, `sidak` : one-step correction
        `hs`, `holm-sidak` : step down method using Sidak adjustments
        `h`, `holm` : step-down method using Bonferroni adjustments
        `sh`, `simes-hochberg` : step-up method  (independent)
        `hommel` : closed method based on Simes tests (non-negative)
        `fdr_i`, `fdr_bh` : Benjamini/Hochberg  (non-negative)
        `fdr_n`, `fdr_by` : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (Benjamini/Hochberg)
        'fdr_tsbky' : two stage fdr correction (Benjamini/Krieger/Yekutieli)
        'fdr_gbs' : adaptive step-down fdr correction (Gavrilov, Benjamini, Sarkar)
    """

    if copy:
        Z = Z.copy()

    if side == "one":
        sideflag = 1
    elif side == "two" or "double":
        sideflag = 2

    Idx = np.triu_indices(Z.shape[0], 1)
    Zv = Z[Idx]

    Pv = stats.norm.cdf(-np.abs(Zv)) * sideflag

    [Hv, adjpvalsv] = smmt.multipletests(Pv, method=mce)[:2]
    adj_pvals = np.zeros(Z.shape)
    Zt = np.zeros(Z.shape)

    Zv[np.invert(Hv)] = 0
    Zt[Idx] = Zv
    Zt = Zt + Zt.T

    adj_pvals[Idx] = adjpvalsv
    adj_pvals = adj_pvals + adj_pvals.T

    adj_pvals[range(Z.shape[0]), range(Z.shape[0])] = 0

    return Zt, binarize(Zt), adj_pvals


class MatManParamError(RuntimeError):
    pass


def binarize(W, copy=True):
    """
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    """
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def invert(W, copy=True):
    """
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        inverted connectivity matrix
    """
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1.0 / W[E]
    return W
