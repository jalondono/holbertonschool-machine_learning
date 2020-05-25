#!/usr/bin/env python3
"""Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
     using batch normalization:"""
    mean = np.mean(Z, axis=0)
    err_std = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(err_std + epsilon)
    Z_norm_end = (gamma * Z_norm) + beta
    return Z_norm_end
