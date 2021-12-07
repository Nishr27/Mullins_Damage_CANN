# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:57:03 2021

@author: nishe
"""
import tensorflow as tf

def P11_stress(stress):
    """
    Convert the P11 stress value into a tensor with diag[P11, 0, 0]

    Parameters
    ----------
    stress : SCALAR
        P11 STRESS VALUES.

    Returns
    -------
    stress_P : THE STRESS TENSOR.

    """
    batch_size = stress.shape[0]
    stress_P = tf.Variable(tf.zeros(shape=(batch_size, 3, 3), dtype=tf.float64))
    
    for i in range(batch_size):
        stress_P[i, 0, 0].assign(stress[i])
    
    return stress_P