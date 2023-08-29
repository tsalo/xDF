#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for xDF.

Created on Fri Mar 16 16:53:44 2018

@author: sorooshafyouni
University of Oxford, 2019
srafyouni@gmail.com
"""
import numpy as np


def AC_fft(Y, T, copy=True):
    """Perform something."""
    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print("AC_fft::: Input should be in IxT form, the matrix was transposed.")
        Y = np.transpose(Y)

    print("AC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y - np.transpose(np.tile(mY2, (T, 1)))

    nfft = int(nextpow2(2 * T - 1))
    # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)
    # be careful with the dimensions
    ACOV = np.real(np.fft.ifft(yfft * np.conj(yfft), axis=1))
    ACOV = ACOV[:, 0:T]

    Norm = np.sum(np.abs(Y) ** 2, axis=1)
    Norm = np.transpose(np.tile(Norm, (T, 1)))
    xAC = ACOV / Norm
    # normalise the COVs

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(T)
    # assumes normality for AC
    CI = [-bnd, bnd]

    return xAC, CI


def xC_fft(Y, T, mxL=[], copy=True):
    """Perform something.

    This should be checked! There shouldn't be any complex numbers!!
    __main__:74: ComplexWarning: Casting complex values to real discards the imaginary part
    This is because Python, in contrast to Matlab, produce highly prcise imaginary parts
    by defualt, when you wanna do ifft, just use np.real()
    """
    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print("xC_fft::: Input should be in IxT form, the matrix was transposed.")
        Y = np.transpose(Y)

    if not np.size(mxL):
        mxL = T

    I = np.shape(Y)[0]

    print("xC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y - np.transpose(np.tile(mY2, (T, 1)))

    nfft = nextpow2(2 * T - 1)
    # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)
    # be careful with the dimensions

    mxLcc = (mxL - 1) * 2 + 1
    xC = np.zeros([I, I, mxLcc])

    XX = np.triu_indices(I, 1)[0]
    YY = np.triu_indices(I, 1)[1]

    for i in np.arange(np.size(XX)):  # loop around edges.
        xC0 = np.fft.ifft(yfft[XX[i], :] * np.conj(yfft[YY[i], :]), axis=0)
        xC0 = np.real(xC0)
        xC0 = np.concatenate((xC0[-mxL + 1 : None], xC0[0:mxL]))

        xC0 = np.fliplr([xC0])[0]
        Norm = np.sqrt(
            np.sum(np.abs(Y[XX[i], :]) ** 2) * np.sum(np.abs(Y[YY[i], :]) ** 2)
        )

        xC0 = xC0 / Norm
        xC[XX[i], YY[i], :] = xC0
        del xC0

    xC = xC + np.transpose(xC, (1, 0, 2))
    lidx = np.arange(-(mxL - 1), mxL)

    return xC, lidx


def nextpow2(x):
    """Returns the first P such that P >= abs(N).

    nextpow2 Next higher power of 2.
    nextpow2(N) returns the first P such that P >= abs(N).
    It is often useful for finding the nearest power of two sequence length for FFT operations.
    """
    return 1 if x == 0 else int(2 ** np.ceil(np.log2(x)))


def tukeytaperme(ac, T, M, verbose=True):
    """Perform single Tukey tapering for given length of window, M, and initial value, intv.

    intv should only be used on crosscorrelation matrices.

    SA, Ox, 2018
    """
    ac = ac.copy()
    # ----Checks:
    if not T in np.shape(ac):
        raise ValueError("tukeytaperme::: There is something wrong, mate!")
        # print('Oi')
    # ----

    M = int(np.round(M))

    tukeymultiplier = (1 + np.cos(np.arange(1, M) * np.pi / M)) / 2
    tt_ts = np.zeros(np.shape(ac))

    if len(np.shape(ac)) == 2:
        if np.shape(ac)[1] != T:
            ac = ac.T
        if verbose:
            print("tukeytaperme::: The input is 2D.")
        N = np.shape(ac)[0]
        tt_ts[:, 0 : M - 1] = np.tile(tukeymultiplier, [N, 1]) * ac[:, 0 : M - 1]

    elif len(np.shape(ac)) == 3:
        if verbose:
            print("tukeytaperme::: The input is 3D.")
        N = np.shape(ac)[0]
        tt_ts[:, :, 0 : M - 1] = (
            np.tile(tukeymultiplier, [N, N, 1]) * ac[:, :, 0 : M - 1]
        )

    elif len(np.shape(ac)) == 1:
        if verbose:
            print("tukeytaperme::: The input is 1D.")
        tt_ts[0 : M - 1] = tukeymultiplier * ac[0 : M - 1]

    return tt_ts


def curbtaperme(ac, T, M, verbose=True):
    """Curb the autocorrelations, according to Anderson 1984.

    multi-dimensional, and therefore is fine!
    SA, Ox, 2018
    """
    ac = ac.copy()
    M = int(round(M))
    msk = np.zeros(np.shape(ac))
    if len(np.shape(ac)) == 2:
        if verbose:
            print("curbtaperme::: The input is 2D.")
        msk[:, 0:M] = 1

    elif len(np.shape(ac)) == 3:
        if verbose:
            print("curbtaperme::: The input is 3D.")
        msk[:, :, 0:M] = 1

    elif len(np.shape(ac)) == 1:
        if verbose:
            print("curbtaperme::: The input is 1D.")
        msk[0:M] = 1

    ct_ts = msk * ac

    return ct_ts


def shrinkme(ac, T):
    """Shrink the *early* bunches of autocorr coefficients beyond the CI.

    Yo! this should be transformed to the matrix form, those fors at the top are bleak!

    SA, Ox, 2018
    """
    ac = ac.copy()

    if np.shape(ac)[1] != T:
        ac = ac.T

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(T)
    # assumes normality for AC

    N = np.shape(ac)[0]
    msk = np.zeros(np.shape(ac))
    BreakPoint = np.zeros(N)
    for i in np.arange(N):
        # finds the break point -- intercept
        TheFirstFalse = np.where(np.abs(ac[i, :]) < bnd)

        # if you couldn't find a break point, then continue = the row will remain zero
        if np.size(TheFirstFalse) == 0:
            continue
        else:
            BreakPoint_tmp = TheFirstFalse[0][0]

        msk[i, :BreakPoint_tmp] = 1
        BreakPoint[i] = BreakPoint_tmp

    return ac * msk, BreakPoint
