# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:56:07 2021

@author: nishe
"""

import tensorflow as tf

def right_CG_tensor(F):
    """
    Calculate the Right Cauchy Green Tensor from the Deformation Gradients.

    Parameters
    ----------
    F : TENSOR
        DEFORMATION GRADIENT (3, 3) TENSOR.

    Returns
    -------
    C : RIGHT CAUCHY GREEN TENSOR

    """
    C = tf.matmul(F, F, transpose_a=True)
    
    return C