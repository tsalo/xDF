# -*- coding: utf-8 -*-
"""The main code for xDF.

Created on Thu Jan 10 13:31:32 2019

@author: sorooshafyouni
University of Oxford, 2019
"""
import logging

import numpy as np
import scipy.stats as sp

from xdf.matrix import correlate_matrix, product_matrix, sum_matrix
from xdf.utils import autocorr_fft, crosscorr_fft, curbtaperme, shrinkme, tukeytaperme

LGR = logging.getLogger("xdf.xdf")


class AutocorrPearson(object):
    """Calculate Pearson correlation and variance matrices accounting for autocorrelation."""

    def __init__(self, method="truncate", methodparam="adaptive", limit_variance=True):
        """Calculate Pearson correlation and variance matrices accounting for autocorrelation.

        Parameters
        ----------
        method : {"tukey", "truncate"}, optional
            The method for estimating autocorrelation.
            Default = "truncate".
        methodparam : :obj:`str`, :obj:`int`, or :obj:`float`, optional
            If ``method`` is "truncate", ``methodparam`` must be "adaptive" or an integer.
            If ``method`` is "tukey", ``methodparam`` must be an empty string ("") or a number.
            Default = "adaptive".
        limit_variance : :obj:`bool`, optional
            If an estimate exceeds the theoretical variance of a white noise then it curbs the
            estimate back to ``(1-rho^2)^2/n_cols``.
            To disable this "curbing", set limit_variance to False.
            Default = True.

        Attributes
        ----------
        correlation_ : :obj:`numpy.ndarray` of shape (n_features, n_features)
            Pearson correlation coefficients.
        variance_ : :obj:`numpy.ndarray` of shape (n_features, n_features)
            Variance of the Pearson correlation coefficients, after correcting for autocorrelation.
            The diagonal is zeroed out.
        varlimit_ : :obj:`float`
            Theoretical variance under x & y are i.i.d; (1-rho^2)^2.
        varlimit_idx_ : :obj:`numpy.ndarray` of shape (K, 2)
            Index of (i,j) edges of which their variance exceeded the theoretical variance.
            K = number of outliers.
        z_corrected_ : :obj:`numpy.ndarray` of shape (n_features, n_features)
            Z test statistics corrected for autocorrelation.
        z_uncorrected_ : :obj:`numpy.ndarray` of shape (n_features, n_features)
            Z test statistics without autocorrelation correction.

        Notes
        -----
        Per :footcite:t:`afyouni2019effective`, method="truncate" + methodparam="adaptive" works
        best.
        """
        self.method = method
        self.methodparam = methodparam
        self.limit_variance = limit_variance

    def fit_transform(self, X):
        """Calculate Pearson correlation and variance matrices accounting for autocorrelation.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples/volumes and
            ``n_features`` is the number of features/voxels/vertices/regions.

        Returns
        -------
        correlation : :obj:`numpy.ndarray` of shape (n_features, n_features)
            Pearson correlation coefficients.
        variance : :obj:`numpy.ndarray` of shape (n_features, n_features)
            Variance of the Pearson correlation coefficients, after correcting for autocorrelation.
            The diagonal is zeroed out.
        """
        dict_ = autocorr_pearson(
            arr=X,
            n_samples=X.shape[0],
            method=self.method,
            methodparam=self.methodparam,
            limit_variance=self.limit_variance,
            copy=True,
        )
        self.correlation_ = dict_["r"]
        self.variance_ = dict_["v"]
        self.variance_z_ = dict_["var_z"]
        self.varlimit_ = dict_["varlimit"]
        self.varlimit_idx_ = dict_["varlimit_idx"]
        self.z_corrected_ = dict_["z"]
        self.z_uncorrected_ = dict_["z_uncorrected"]

        return self.correlation_, self.variance_


def autocorr_pearson(
    arr,
    n_samples,
    method="truncate",
    methodparam="adaptive",
    limit_variance=True,
    copy=True,
):
    """Run xDF.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray` of shape (V, T)
        Time series array to correlate with xDF.
        V = number of features/regions/voxels
        T = number of samples/data points/volumes
    n_samples : :obj:`int`
        Number of data points. Should match dimension 0 of ``arr``.
    method : {"tukey", "truncate"}
    methodparam : :obj:`str`, :obj:`int`, or :obj:`float`
        If ``method`` is "truncate", ``methodparam`` must be "adaptive" or an integer.
        If ``method`` is "tukey", ``methodparam`` must be an empty string ("") or a number.
        If ``methodparam`` is an empty string, then a default value of sqrt(n_samples) will
        be used, as recommended in :footcite:t:`chatfield2016analysis`.
    limit_variance : :obj:`bool`, optional
        If an estimate exceeds the theoretical variance of a white noise then it curbs the
        estimate back to (1-rho^2)^2/n_features.
        To disable this "curbing", set limit_variance to False.
        Default = True.
    copy : :obj:`bool`, optional
        If False, this function may modify the original data array.
        Default = True.

    Returns
    -------
    out : :obj:`dict`
        A dictionary containing the following keys:
        -   "p": IxI array of uncorrected p-values.
        -   "z": IxI array of z-scores, adjusted for autocorrelation.
        -   "z_uncorrected": IxI array of z-scores without any autocorrelation adjustment.
        -   "v": IxI array of variance of correlation coefficient between corresponding elements,
            with the diagonal set to 0.
        -   "var_z": IxI array of variance of z-transformed correlation coefficient between
            corresponding elements, with the diagonal set to 0.
        -   "varlimit": Theoretical variance under x & y are i.i.d; (1-rho^2)^2.
        -   "varlimit_idx": Index of (i,j) edges of which their variance exceeded the theoretical
            variance.

    Notes
    -----
    Per :footcite:t:`afyouni2019effective`, method="truncate" + methodparam="adaptive" works best.
    """
    # Make sure you are not messing around with the original time series
    if copy:
        arr = arr.copy()

    arr = arr.T
    if arr.shape[1] != n_samples:
        assert arr.shape[0] == n_samples
        LGR.debug("Input should be in (n_samples, n_features) form, the matrix was transposed.")
        arr = arr.T

    n_rows = arr.shape[0]

    # standardise time series by standard deviations
    ts_std = np.std(arr, axis=1, ddof=1)
    arr = arr / np.tile(ts_std, (n_samples, 1)).T
    LGR.info("Time series standardised by their standard deviations.")

    # Estimate xC and AC
    # Corr
    rho, z_uncorrected = correlate_matrix(arr, n_samples)
    rho = np.round(rho, 7)
    z_uncorrected = np.round(z_uncorrected, 7)

    # Autocorr
    ac, _ = autocorr_fft(arr, n_samples)
    ac = ac[:, 1 : n_samples - 1]
    # The last element of ACF is rubbish, the first one is 1, so why bother?!
    nLg = n_samples - 2

    # Cross-corr
    xcf, _ = crosscorr_fft(arr, n_samples)

    xc_p = xcf[:, :, 1 : n_samples - 1]
    xc_p = np.flip(xc_p, axis=2)
    # positive-lag xcorrs
    xc_n = xcf[:, :, n_samples:-1]
    # negative-lag xcorrs

    # Start of Regularisation
    if method.lower() == "tukey":
        # The np.sqrt(n_samples) value is suggested in Chatfield 2016.
        M = np.sqrt(n_samples) if not methodparam else methodparam

        LGR.debug(f"AC Regularisation: Tukey tapering of M = {int(np.round(M))}")
        ac = tukeytaperme(ac, nLg, M)
        xc_p = tukeytaperme(xc_p, nLg, M)
        xc_n = tukeytaperme(xc_n, nLg, M)

    elif method.lower() == "truncate":
        # Adaptive Truncation
        if isinstance(methodparam, str):
            if methodparam.lower() != "adaptive":
                raise ValueError(
                    "What?! Choose adaptive as the option, or pass an integer for truncation"
                )

            LGR.debug("AC Regularisation: Adaptive Truncation")
            ac, bp = shrinkme(ac, nLg)

            # truncate the cross-correlations, by the breaking point found from the ACF.
            # (choose the largest of two)
            for i_row in np.arange(n_rows):
                for j_row in np.arange(n_rows):
                    maxBP = np.max([bp[i_row], bp[j_row]])
                    xc_p[i_row, j_row, :] = curbtaperme(ac=xc_p[i_row, j_row, :], M=maxBP)
                    xc_n[i_row, j_row, :] = curbtaperme(ac=xc_n[i_row, j_row, :], M=maxBP)

        elif isinstance(methodparam, int):  # Npne-Adaptive Truncation
            LGR.debug(f"AC Regularisation: Non-adaptive Truncation on M = {methodparam}")
            ac = curbtaperme(ac=ac, M=methodparam)
            xc_p = curbtaperme(ac=xc_p, M=methodparam)
            xc_n = curbtaperme(ac=xc_n, M=methodparam)

        else:
            raise ValueError("methodparam for truncation method should be either str or int.")

    # Start of Regularisation

    # Start of the Monster Equation
    wgt = np.arange(nLg, 0, -1)
    wgtm2 = np.tile((np.tile(wgt, [n_rows, 1])), [n_rows, 1])
    wgtm3 = np.reshape(wgtm2, [n_rows, n_rows, np.size(wgt)])
    # this is shit, eats all the memory!
    Tp = n_samples - 1

    # Da Equation!--------------------
    var = (
        Tp * (1 - rho**2) ** 2
        + rho**2 * np.sum(wgtm3 * (sum_matrix(ac**2, nLg) + xc_p**2 + xc_n**2), axis=2)
        - 2 * rho * np.sum(wgtm3 * (sum_matrix(ac, nLg) * (xc_p + xc_n)), axis=2)
        + 2 * np.sum(wgtm3 * (product_matrix(ac, nLg) + (xc_p * xc_n)), axis=2)
    ) / (n_samples**2)
    # End of the Monster Equation

    # Truncate to Theoritical Variance
    varlimit = (1 - rho**2) ** 2 / n_samples
    np.fill_diagonal(varlimit, 0)

    varlimit_idx = np.where(var < varlimit)
    n_var_outliers = varlimit_idx[1].size / 2

    if n_var_outliers > 0 and limit_variance:
        LGR.debug("Variance truncation is ON.")

        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        var[varlimit_idx] = varlimit[varlimit_idx]

        FGE = (n_rows * (n_rows - 1)) / 2
        LGR.debug(
            f"{n_var_outliers} ({str(round((n_var_outliers / FGE) * 100, 3))}%) "
            "edges had variance smaller than the textbook variance!"
        )
    else:
        LGR.debug("NO truncation to the theoritical variance.")

    # Start of Statistical Inference

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    # delta method; make sure the N is correct! So they cancel out.
    var_z = var / ((1 - rho**2) ** 2)
    z_corrected = rf / np.sqrt(var_z)
    p_corrected = 2 * sp.norm.cdf(-abs(z_corrected))  # both tails

    # diagonal is rubbish
    np.fill_diagonal(var, 0)
    np.fill_diagonal(var_z, 0)
    # NaN screws up everything, so get rid of the diag, but be careful here.
    np.fill_diagonal(p_corrected, 0)
    np.fill_diagonal(z_corrected, 0)

    # End of Statistical Inference
    out = {
        "r": rho,
        "p": p_corrected,
        "z": z_corrected,
        "z_uncorrected": z_uncorrected,
        "v": var,
        "var_z": var_z,
        "varlimit": varlimit,
        "varlimit_idx": varlimit_idx,
    }

    return out
