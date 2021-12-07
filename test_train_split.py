# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:07:12 2021

@author: nishe
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split

def train_test(data):
    """
    Splitting the data into test and train sets.

    Parameters
    ----------
    data : Array-like
        Deformation gradient, loading or stress data.

    Returns
    -------
    data_train, data_test

    """
    data_train, data_test = train_test_split(data.numpy(), test_size=0.2, random_state=42)
    
    return data_train, data_test    