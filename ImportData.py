# Methods to import data
import  keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint as mc
import h5py

def update():
    print(" in Update")


# Define weight history function to keep track of weights over iterations
class WeightHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.w1_weights = []
        self.w2_weights = []
        self.w3_weights = []
        self.bias_weights = []
        self.iteration_count = []
        self.accuracy_train = []

    def on_batch_end(self, batch, logs={}):
        self.bias_weights.append(model.layers[2].get_weights()[1][0])  # bias
        self.w1_weights.append(model.layers[2].get_weights()[0][0][0])  # w1
        self.w2_weights.append(model.layers[2].get_weights()[0][1][0])  # w2
        self.w3_weights.append(model.layers[2].get_weights()[0][2][0])  # w3
        self.iteration_count.append(len(self.bias_weights))  # iterations
        self.accuracy_train.append(logs.get('acc'))
