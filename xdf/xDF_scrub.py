#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:31:32 2019

@author: sorooshafyouni
University of Oxford, 2019
"""

import itertools

import numpy as np
import scipy.signal as ss
import scipy.stats as sp
from scipy.linalg import toeplitz

from xdf.utils import curbtaperme, tukeytaperme, shrinkme


def autocorr_pearson(
    arr,
    n_samples,
    method="truncate",
    methodparam="adaptive",
    limit_variance=True,
    copy=True,
):
    """Calculate the xDF matrix for a given time series.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray` of shape (V, n_samples)
        Time series array to correlate with xDF.
        V = number of features/regions/voxels
        n_samples = number of samples/data points/volumes
    n_samples : :obj:`int`
        Number of data points/volumes. Should match dimension 0 of ``arr``.
    method : {"tukey", "truncate"}, optional
        The method for estimating autocorrelation.
        Default = "truncate".
    methodparam : :obj:`str`, :obj:`int`, or :obj:`float`, optional
        If ``method`` is "truncate", ``methodparam`` must be "adaptive" or an integer.
        If ``method`` is "tukey", ``methodparam`` must be an empty string ("") or a number.
        Default = "adaptive".
    limit_variance : bool, optional
        If an estimate is lower than the theoretical variance of a white noise then it increases the
        estimate up to ``(1-rho^2)^2/n_cols``.
        To disable this "curbing", set limit_variance to False.
        Default = True.
    copy : bool, optional
        If False, this function may modify the original data array.
        Default = True.

    Returns
    -------
    var_hat_rho : array-like of shape (n_rows, n_rows)
        Variance of correlation coefficient between corresponding elements,
        with the diagonal set to 0.
    """
    if copy:
        arr = arr.copy()

    if np.shape(arr)[1] != n_samples:
        print("Second dimension should be n_samples; the matrix was transposed")
        arr = np.transpose(arr)

    n_rows = np.shape(arr)[0]

    scrubbed_frames = np.where(np.any(np.isnan(arr), axis=0))[0]
    n_retained_samples = n_samples - len(scrubbed_frames)

    arr = arr - np.nanmean(arr, axis=1)[:, None]
    norms = np.nanmean(arr**2, axis=1)[:, None]

    # Mask NaNs with 0s for summation
    if n_retained_samples < n_samples:
        arr[:, scrubbed_frames] = 0

    # Calculate denominator according to acf in R
    list_pairs = list(
        itertools.combinations(np.setdiff1d(np.arange(n_samples), scrubbed_frames), 2)
    )
    n_pairs = [n_retained_samples] + list(np.bincount([y - x for (x, y) in list_pairs], minlength=n_samples)[1:])
    n_denom = n_pairs + np.arange(n_samples)

    # Calculate autocorrelation
    ac = (
        np.concatenate(
            [
                ss.correlate(arr[i, :], arr[i, :], method="fft")[None, -n_samples:] / n_denom
                for i in np.arange(n_rows)
            ],
            axis=0,
        )
        / norms
    )

    # Calculate cross-correlation
    xc = np.zeros((n_rows, n_rows, 2 * n_samples - 1))
    triu_idx_x = np.triu_indices(n_rows, 1)[0]
    triu_idx_y = np.triu_indices(n_rows, 1)[1]

    for i, j in zip(triu_idx_x, triu_idx_y):
        xc[i, j, :] = ss.correlate(arr[i, :], arr[j, :], method="fft") / (
            np.sqrt(norms[i] * norms[j])
            * np.concatenate((np.flip(np.delete(n_denom, 0)), n_denom))
        )

    xc = xc + np.transpose(xc, (1, 0, 2))

    # Extract positive and negative cross-correlations
    xc_p = xc[:, :, : (n_samples - 1)]
    xc_p = np.flip(xc_p, axis=2)
    xc_n = xc[:, :, -(n_samples - 1) :]

    # Extract lag-0 correlations
    rho = np.eye(n_rows) + xc[:, :, n_samples - 1]

    # Regularize!
    if method.lower() == "tukey":
        if methodparam == "":
            M = np.sqrt(n_samples)
        else:
            M = methodparam

        print(f"AC regularization: Tukey tapering of M = {int(np.round(M))}")

        ac = tukeytaperme(ac, n_samples - 1, M)
        xc_p = tukeytaperme(xc_p, n_samples - 1, M)
        xc_n = tukeytaperme(xc_n, n_samples - 1, M)

    elif method.lower() == "truncate":
        if isinstance(methodparam, str):  # Adaptive truncation
            if methodparam.lower() != "adaptive":
                raise ValueError("methodparam for truncation must be 'adaptive' or an integer")

            print("AC regularization: adaptive truncation")

            ac, bp = shrinkme(ac, n_samples)

            for i in np.arange(n_rows):
                for j in np.arange(n_rows):
                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = curbtaperme(
                        xc_p[i, j, :], n_samples - 1, maxBP,
                        verbose=False,
                    )
                    xc_n[i, j, :] = curbtaperme(
                        xc_n[i, j, :], n_samples - 1, maxBP,
                        verbose=False,
                    )

        elif isinstance(methodparam, int):  # Non-adaptive truncation
            print(f"AC regularization: non-adaptive truncation on M = {methodparam}")
            ac = curbtaperme(ac, n_samples - 1, methodparam)
            xc_p = curbtaperme(xc_p, n_samples - 1, methodparam)
            xc_n = curbtaperme(xc_n, n_samples - 1, methodparam)

        else:
            raise ValueError("methodparam for truncation method should be either str or int")
    else:
        raise ValueError("Method parameter must be either 'tukey' or 'truncate'.")

    # Estimate variance (big formula)
    var_hat_rho = np.zeros((n_rows, n_rows))

    for i, j in zip(triu_idx_x, triu_idx_y):
        r = rho[i, j]

        Sigx = toeplitz(ac[i, :])
        Sigy = toeplitz(ac[j, :])

        Sigx = np.delete(Sigx, scrubbed_frames, axis=0)
        Sigx = np.delete(Sigx, scrubbed_frames, axis=1)
        Sigy = np.delete(Sigy, scrubbed_frames, axis=0)
        Sigy = np.delete(Sigy, scrubbed_frames, axis=1)

        Sigxy = np.triu(toeplitz(np.insert(xc_p[i, j, :], 0, r)), k=1) + np.tril(
            toeplitz(np.insert(xc_n[i, j, :], 0, r))
        )
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=0)
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=1)
        Sigyx = np.transpose(Sigxy)

        var_hat_rho[i, j] = (
            (r**2 / 2) * np.trace(Sigx @ Sigx)
            + (r**2 / 2) * np.trace(Sigy @ Sigy)
            + r**2 * np.trace(Sigyx @ Sigxy)
            + np.trace(Sigxy @ Sigxy)
            + np.trace(Sigx @ Sigy)
            - 2 * r * np.trace(Sigx @ Sigxy)
            - 2 * r * np.trace(Sigy @ Sigyx)
        ) / n_retained_samples**2

    var_hat_rho = var_hat_rho + np.transpose(var_hat_rho)

    # Truncate to theoretical variance
    varlimit = (1 - rho**2) ** 2 / n_retained_samples

    varlimit_idx = np.where(var_hat_rho < varlimit)
    n_var_outliers = varlimit_idx[1].size / 2
    if n_var_outliers > 0 and limit_variance:
        print("Variance truncation is ON.")

        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        var_hat_rho[varlimit_idx] = varlimit[varlimit_idx]

        FGE = (n_rows * (n_rows - 1)) / 2
        print(
            f"{n_var_outliers} ({str(round((n_var_outliers / FGE) * 100, 3))}%) "
            "edges had variance smaller than the textbook variance!"
        )
    else:
        print("NO truncation to the theoretical variance.")

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    # delta method; make sure the N is correct! So they cancel out.
    var_z = var_hat_rho / ((1 - rho**2) ** 2)
    z_corrected = rf / np.sqrt(var_z)
    p_corrected = 2 * sp.norm.cdf(-abs(z_corrected))  # both tails
    z_uncorrected = rf / np.sqrt(n_retained_samples - 3)

    # diagonal is rubbish
    np.fill_diagonal(var_hat_rho, 0)
    np.fill_diagonal(var_z, 0)
    # NaN screws up everything, so get rid of the diag, but be careful here.
    np.fill_diagonal(p_corrected, 0)
    np.fill_diagonal(z_corrected, 0)

    out = {
        "r": rho,
        "p": p_corrected,
        "z": z_corrected,
        "z_uncorrected": z_uncorrected,
        "v": var_hat_rho,
        "var_z": var_z,
        "varlimit": varlimit,
        "varlimit_idx": varlimit_idx,
    }

    return out
