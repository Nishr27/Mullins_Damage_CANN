# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:19:21 2021

@author: nishe
"""
import tensorflow as tf
from tensorflow import keras

from right_CG_tensor import right_CG_tensor
from first_invariant import first_invariant
from second_invariant import second_invariant

def first_PK(x):
    F = x[0]
    S = x[1]
    return tf.matmul(F, S)
    
def CANN_model():
    """
    Preparing the Neural Network Model.

    Returns
    -------
    A functional Neural Network Model that takes as inputs Deformation gradients and loads and outputs the stress.

    """
    F        = keras.layers.Input(shape=(3, 3), dtype=tf.float64, name="F")
    load     = keras.layers.Input(shape=(1,), dtype=tf.float64, name="Load")
    
    C        = keras.layers.Lambda(right_CG_tensor, name="C")(F)  
    I        = keras.layers.Lambda(first_invariant, name="I")(C)
    II       = keras.layers.Lambda(second_invariant, name="II")(C)
    concat   = keras.layers.concatenate([I, II, load], name="Concatenate")
    
    hidden0  = keras.layers.Dense(10, activation="sigmoid", name="hidden_layer")(concat)
    
    Psi      = keras.layers.Dense(1, activation="sigmoid", name="Strain_Energy")(hidden0)
    S        = keras.layers.Lambda(lambda x: tf.gradients(x[0], x[1], unconnected_gradients=tf.UnconnectedGradients.NONE), name="S")(Psi, C)
    P        = keras.layers.Lambda(first_PK, output_shape=(3, 3))(F, S)
    stress   = keras.layers.Dense(3, activation="relu", name="Stress")(P)
    
    model = keras.models.Model(inputs=[F, load], outputs=stress)
    
    return model