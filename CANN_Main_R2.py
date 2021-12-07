# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:34:40 2021

@author: nishe
"""
#%%
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
from stress_strain_plot import stress_strain_plot
from input_prep import input_prep
from test_train_split import train_test
from P11_stress import P11_stress
#%%
from CANN_model import CANN_model

#%%
# Importing the prepared Mullins Data
mullins_data = pd.read_csv("C:\My Files\Programs\CANN\Mullins_Data.csv", names=["Strain", "Stress"])
print(mullins_data.head(5))

# Separating the Stress Strain Data
stress = tf.Variable(mullins_data["Stress"])
strain = tf.Variable(mullins_data["Strain"])


# Visualizing the data 
stress_strain_plot(stress, strain)

# Preparing the data as input
area = tf.constant(20, dtype=tf.float64) # area in mm^2
F, loading = input_prep(stress, strain, area, loading_para=1)

#Setting the random seed
tf.random.set_seed(42)
#%%
# Splitting the data into train and test sets
F_train, F_test             = train_test(F)
loading_train, loading_test = train_test(loading)
stress_train, stress_test   = train_test(stress)
stress_train_P, stress_test_P = P11_stress(stress_train), P11_stress(stress_test)
#%%
strain_train, strain_test   = train_test(strain) 
#%%
print(loading_train)
#%%
print(stress_train_P)
#%%
# Defining CANN model
model = CANN_model()
model.summary()

#%%

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(loss="mse", optimizer=opt)

# Train the neural network
history = model.fit([F_train, loading_train], stress_train_P.numpy(), validation_data=([F_test, loading_test], stress_test_P.numpy()), epochs=1000)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

#%%
# Test the model on the test set
stress_pred = model.predict([F_test, loading_test])

# Compare the predictions with the actual
print(f"The predictions are: {stress_pred}")
print(f"\n The actual stresses are: {stress_test}")
#%%
# Graphing the results
plt.scatter(stress_pred, strain_test, c="r")
plt.scatter(stress_test, strain_test, c="y")