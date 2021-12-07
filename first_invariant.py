# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:51:45 2021

@author: nishe
"""
import tensorflow as tf

def first_invariant(C):
    """
    Calculating the first Strain-Invariant from the right Cauchy-Green Tensor.

    Parameters
    ----------
    C : TENSOR
        THE RIGHT CAUCHY GREEN TENSOR

    Returns
    -------
    I : THE FIRST STRAIN INVARIANT

    """
    I = tf.linalg.trace(C)
    
    return tf.reshape(I, shape=(-1, 1))