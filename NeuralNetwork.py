import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint as mc
import h5py
import matplotlib.pyplot as plt


# Starting configs

model = Sequential()
model.add(Dense(3, activation='sigmoid', input_dim=8))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
