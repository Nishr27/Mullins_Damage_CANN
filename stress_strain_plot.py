# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:24:03 2021

@author: nishe
"""
import matplotlib.pyplot as plt

def stress_strain_plot(stress, strain):
    """
    Generating the plot of Stress vs Strain

    """
    plt.plot(strain.numpy(), stress.numpy(), marker="s", linewidth=2, markersize=4)
    plt.xlabel("Strain")
    plt.ylabel("Stress in MPa")
    plt.title("Stress vs Strain")
    plt.show()