#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demonstrate the xDF method.

Created on Tue Jan 15 11:56:37 2019

@author: sorooshafyouni
University of Oxford, 2019
"""
import numpy as np
import scipy.io

from xdf.xdf import xDF_Calc
from xdf.matrix import stat_threshold

if __name__ == "__main__":
    V = "/Users/sorooshafyouni/Home/BCF/BCFAnal/FC/100HCPTimeSeries/Yeo/HCP_FPP_124422_OnlyMTS.mat"
    mat = scipy.io.loadmat(V)
    mts = mat["mts"]
    T = 1200

    print("+++++++ xDF without regularisation::: +++++++++++++++++++++++++++++++")
    xDFOut_TVOn = xDF_Calc(mts, T, method="", TV=True)

    Z = stat_threshold(xDFOut_TVOn["z"], mce="b")[0]
    print(len(np.where(Z != 0)[0]) / 2)

    print("+++++++ xDF without truncation::: +++++++++++++++++++++++++++++++++++")
    xDFOut_tna = xDF_Calc(mts, T, method="truncate", methodparam=int(T / 4), verbose=True)

    Z = stat_threshold(xDFOut_tna["z"], mce="b")[0]
    print(len(np.where(Z != 0)[0]) / 2)

    print("+++++++ xDF with ADAPTIVE truncation::: ++++++++++++++++++++++++++++++")
    xDFOut_ta = xDF_Calc(mts, T, method="truncate", methodparam="adaptive", verbose=True)
    Z = stat_threshold(xDFOut_ta["z"], mce="fdr_bh")[0]
    print(len(np.where(Z != 0)[0]) / 2)

    print("+++++++ xDF without tapering::: +++++++++++++++++++++++++++++++++++++")
    xDFOut_tt = xDF_Calc(mts, T, method="tukey", methodparam=np.sqrt(T), verbose=True)

    Z = stat_threshold(xDFOut_tt["z"], mce="b")[0]
    print(len(np.where(Z != 0)[0]) / 2)
