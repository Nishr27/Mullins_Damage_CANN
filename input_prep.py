# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:32:23 2021

@author: nishe
"""
import tensorflow as tf

def input_prep(stress, strain, area, loading_para=1):
    """
    Takes the stress and strain inputs and provides the load and deformation gradient values.
    Also, takes in the cross-sectional area and type of loading parameter.

    Parameters
    ----------
    stress       : The P11 Stress or the first Piola Kirchhoff Stress.
    strain       : The strain produced in the model.
    area         : The cross sectional area.
    loading_para : The loading method - 
                        1. Uniaxial Tension
                        2. Equibiaxial Tension
                        3. Pure Shear

    Returns
    -------
    def_grad    : The deformation gradients.
    loading     : The loading values. 

    """
    # Stress = Force/Area
    loading = stress/area
    
    batch_size = stress.shape[0]
    stretch = strain + 1
    F = tf.Variable(tf.zeros((batch_size, 3, 3), dtype=tf.float64))
    
    if loading_para == 1:
        for i in range(batch_size):
            F[i, 0, 0].assign(stretch[i])
            F[i, 1, 1].assign(1/tf.sqrt(stretch[i]))
            F[i, 2, 2].assign(1/tf.sqrt(stretch[i]))
    
    elif loading_para == 2:
        for i in range(batch_size):
            F[i, 0, 0].assign(1/tf.square(stretch[i]))
            F[i, 1, 1].assign(stretch[i])
            F[i, 2, 2].assign(stretch[i])
    
    elif loading_para == 3:
        for i in range(batch_size):
            F[i, 0, 0].assign(stretch[i])
            F[i, 1, 1].assign(1/stretch[i])
            F[i, 2, 2].assign(tf.float64(1.0))
    
    return F, loading