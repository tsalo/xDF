# -*- coding: utf-8 -*-
"""Matrix operations for xDF.

@author: sorooshafyouni
University of Oxford, 2019
"""
import logging

import numpy as np

LGR = logging.getLogger("xdf.matrix")


def sum_matrix(arr, n_rows, copy=True):
    """Perform element-wise summation of each row with other rows.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray`
        a 2D matrix of size TxN
    n_rows : :obj:`int`
        Expected number of rows in ``arr``.
        If the number of rows doesn't match, then ``arr`` will be transposed.

    Returns
    -------
    out : :obj:`numpy.ndarray`
        3D matrix, obtained from element-wise summation of each row with other rows.

    Notes
    -----
    SA, Ox, 2019
    """
    if copy:
        arr = arr.copy()

    if arr.shape[0] != n_rows:
        assert arr.shape[1] == n_rows
        LGR.info("Input should be in TxN form, the matrix was transposed.")
        arr = arr.T

    n_cols = arr.shape[1]
    idx = np.triu_indices(n_cols)
    out = np.empty([n_cols, n_cols, n_rows])
    for i_col in range(idx[0].size - 1):
        xx = idx[0][i_col]
        yy = idx[1][i_col]
        out[xx, yy, :] = arr[:, xx] + arr[:, yy]
        out[yy, xx, :] = arr[:, yy] + arr[:, xx]

    return out


def product_matrix(arr, n_rows, copy=True):
    """Perform element-wise multiplication of each row with other rows.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray`
        a 2D matrix of size TxN
    n_rows : :obj:`int`
        Expected number of rows in ``arr``.
        If the number of rows doesn't match, then ``arr`` will be transposed.

    Returns
    -------
    out : :obj:`numpy.ndarray`
        3D matrix, obtained from element-wise multiplication of each row with other rows.

    Notes
    -----
    SA, Ox, 2019
    """
    if copy:
        arr = arr.copy()

    if arr.shape[0] != n_rows:
        assert arr.shape[1] == n_rows
        LGR.info("Input should be in TxN form, the matrix was transposed.")
        arr = arr.T

    n_cols = arr.shape[1]
    idx = np.triu_indices(n_cols)
    out = np.empty([n_cols, n_cols, n_rows])
    for i in range(idx[0].size - 1):
        xx = idx[0][i]
        yy = idx[1][i]
        out[xx, yy, :] = arr[:, xx] * arr[:, yy]
        out[yy, xx, :] = arr[:, yy] * arr[:, xx]

    return out


def correlate_matrix(arr, n_cols, copy=True):
    """Produce sample correlation matrix and naively corrected z matrix.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray`
        a 2D matrix of size TxN
    n_cols : :obj:`int`
        Expected number of columns in ``arr``.
        If the number of rows doesn't match, then ``arr`` will be transposed.

    Returns
    -------
    r_arr : :obj:`numpy.ndarray`
        Correlation coefficient matrix.
    z_arr : :obj:`numpy.ndarray`
        Z-statistic (as in test statistic) matrix.

    Notes
    -----
    This function zeros out the diagonals of both output arrays.

    SA, Ox, 2019
    """
    if copy:
        arr = arr.copy()

    if arr.shape[1] != n_cols:
        assert arr.shape[1] == n_cols
        LGR.info("Input should be in IxT form, the matrix was transposed.")
        arr = arr.T

    r_arr = np.corrcoef(arr)
    z_arr = np.arctanh(r_arr) * np.sqrt(n_cols - 3)

    np.fill_diagonal(r_arr, 0)
    np.fill_diagonal(z_arr, 0)

    return r_arr, z_arr
