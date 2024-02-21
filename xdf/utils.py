# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for xDF.

Created on Fri Mar 16 16:53:44 2018

@author: sorooshafyouni
University of Oxford, 2019
srafyouni@gmail.com
"""
import logging

import numpy as np

LGR = logging.getLogger("xdf.utils")


def autocorr_fft(arr, n_cols, copy=True):
    """Estimate autocorrelations using fast Fourier transform.

    Parameters
    ----------
    arr
    n_cols
    copy

    Returns
    -------
    xAC : numpy.ndarray of shape (?, ?)
        Autocorrelation matrix.
    ci : list of length 2
        Lower and upper bounds of the confidence interval.
        What is the interval though? 95%?
    """
    if copy:
        arr = arr.copy()

    if arr.shape[1] != n_cols:
        assert arr.shape[0] == n_cols
        LGR.info("Input should be in IxT form, the matrix was transposed.")
        arr = arr.T

    LGR.info("Demean along T")
    arr -= np.mean(arr, axis=1, keepdims=True)

    nfft = int(nextpow2(2 * n_cols - 1))
    # zero-pad the hell out!
    yfft = np.fft.fft(arr, n=nfft, axis=1)
    # be careful with the dimensions
    ACOV = np.real(np.fft.ifft(yfft * np.conj(yfft), axis=1))
    ACOV = ACOV[:, 0:n_cols]

    Norm = np.sum(np.abs(arr) ** 2, axis=1)
    Norm = np.tile(Norm, (n_cols, 1)).T
    xAC = ACOV / Norm
    # normalise the COVs

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(n_cols)
    # assumes normality for AC
    ci = [-bnd, bnd]

    return xAC, ci


def crosscorr_fft(arr, n_cols, mxL=[], copy=True):
    """Estimate crosscorrelations using fast Fourier transform.

    Parameters
    ----------
    arr
    n_cols
    copy

    Returns
    -------
    xAC
    CI

    Notes
    -----
    This should be checked! There shouldn't be any complex numbers!!

    __main__:74: ComplexWarning: Casting complex values to real discards the imaginary part
    This is because Python, in contrast to Matlab, produce highly prcise imaginary parts
    by defualt, when you wanna do ifft, just use np.real()
    """
    if copy:
        arr = arr.copy()

    if np.shape(arr)[1] != n_cols:
        assert arr.shape[0] == n_cols
        LGR.info("Input should be in IxT form, the matrix was transposed.")
        arr = arr.T

    if not np.size(mxL):
        mxL = n_cols

    n_rows = arr.shape[0]

    LGR.info("Demean along T")
    mY2 = np.mean(arr, axis=1)
    arr = arr - np.tile(mY2, (n_cols, 1)).T

    nfft = nextpow2(2 * n_cols - 1)
    # zero-pad the hell out!
    yfft = np.fft.fft(arr, n=nfft, axis=1)
    # be careful with the dimensions

    mxLcc = (mxL - 1) * 2 + 1
    xC = np.zeros([n_rows, n_rows, mxLcc])

    XX = np.triu_indices(n_rows, 1)[0]
    YY = np.triu_indices(n_rows, 1)[1]

    for i in np.arange(np.size(XX)):  # loop around edges.
        xC0 = np.fft.ifft(yfft[XX[i], :] * np.conj(yfft[YY[i], :]), axis=0)
        xC0 = np.real(xC0)
        xC0 = np.concatenate((xC0[-mxL + 1 : None], xC0[0:mxL]))

        xC0 = np.fliplr([xC0])[0]
        Norm = np.sqrt(np.sum(np.abs(arr[XX[i], :]) ** 2) * np.sum(np.abs(arr[YY[i], :]) ** 2))

        xC0 = xC0 / Norm
        xC[XX[i], YY[i], :] = xC0
        del xC0

    xC = xC + np.transpose(xC, (1, 0, 2))
    lidx = np.arange(-(mxL - 1), mxL)

    return xC, lidx


def nextpow2(x):
    """Return the first P such that P >= abs(N).

    Parameters
    ----------
    x

    Returns
    -------
    :obj:`int`

    Notes
    -----
    nextpow2 Next higher power of 2.
    nextpow2(N) returns the first P such that P >= abs(N).
    It is often useful for finding the nearest power of two sequence length for FFT operations.
    """
    return 1 if x == 0 else int(2 ** np.ceil(np.log2(x)))


def tukeytaperme(ac, T, M):
    """Perform single Tukey tapering for given length of window, M, and initial value, intv.

    Parameters
    ----------
    ac
    T
    M

    Returns
    -------
    tt_ts

    Notes
    -----
    intv should only be used on crosscorrelation matrices.

    SA, Ox, 2018
    """
    if T not in ac.shape:
        raise ValueError("There is something wrong, mate!")

    ac = ac.copy()

    M = int(np.round(M))

    tukeymultiplier = (1 + np.cos(np.arange(1, M) * np.pi / M)) / 2
    tt_ts = np.zeros(ac.shape)

    if ac.ndim == 2:
        LGR.debug("The input is 2D.")
        if ac.shape[1] != T:
            ac = ac.T

        n_rows = ac.shape[0]
        tt_ts[:, : M - 1] = np.tile(tukeymultiplier, [n_rows, 1]) * ac[:, : M - 1]

    elif ac.ndim == 3:
        LGR.debug("The input is 3D.")

        n_rows = ac.shape[0]
        tt_ts[:, :, : M - 1] = (
            np.tile(
                tukeymultiplier,
                [n_rows, n_rows, 1],
            )
            * ac[:, :, : M - 1]
        )

    elif ac.ndim == 1:
        LGR.debug("The input is 1D.")

        tt_ts[: M - 1] = tukeymultiplier * ac[: M - 1]

    return tt_ts


def curbtaperme(ac, M):
    """Curb the autocorrelations, according to Anderson 1984.

    Parameters
    ----------
    ac
    M

    Returns
    -------
    ct_ts

    Notes
    -----
    multi-dimensional, and therefore is fine!
    SA, Ox, 2018
    """
    ac = ac.copy()
    M = int(round(M))
    msk = np.zeros(np.shape(ac))
    if ac.ndim == 2:
        LGR.debug("The input is 2D.")
        msk[:, :M] = 1

    elif ac.ndim == 3:
        LGR.debug("The input is 3D.")
        msk[:, :, :M] = 1

    elif ac.ndim == 1:
        LGR.debug("The input is 1D.")
        msk[:M] = 1

    ct_ts = msk * ac

    return ct_ts


def shrinkme(ac, T):
    """Shrink the *early* bunches of autocorr coefficients beyond the CI.

    Parameters
    ----------
    ac
    T

    Returns
    -------
    masked_ac
    BreakPoint

    Notes
    -----
    Yo! this should be transformed to the matrix form, those fors at the top are bleak!

    SA, Ox, 2018
    """
    ac = ac.copy()

    if np.shape(ac)[1] != T:
        ac = ac.T

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(T)
    # assumes normality for AC

    n_rows = np.shape(ac)[0]
    msk = np.zeros(np.shape(ac))
    BreakPoint = np.zeros(n_rows)
    for i in np.arange(n_rows):
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
