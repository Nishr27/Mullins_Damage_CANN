# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:00:57 2021

@author: nishe
"""
import tensorflow as tf

def second_invariant(x):
    """
    Calculating the second invariant from the right Cauchy Green Tensor.

    Parameters
    ----------
    C : TENSOR
        RIGHT CAUCHY GREEN TENSOR.

    Returns
    -------
    II : THE SECOND STRAIN INVARIANT

    """
    II = 0.5* (tf.math.square(tf.linalg.trace(x)) - (tf.linalg.trace(tf.matmul(x, x, transpose_a=True))))
    
    return tf.reshape(II, shape=(-1, 1))