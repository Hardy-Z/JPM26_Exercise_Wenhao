import tensorflow as tf
from tensorflow import keras
from config import *

class PolicyNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = keras.layers.Dense(Config.HIDDEN_UNITS, activation='relu')
        self.fc2 = keras.layers.Dense(Config.HIDDEN_UNITS, activation='relu')
        self.out = keras.layers.Dense(1, activation='linear')

    def call(self, x):
        # x shape: (batch, 2) = [k, z]
        tf.debugging.assert_all_finite(x[:,1], "z has NaN/Inf")
        tf.debugging.assert_all_finite(x[:,0], "k has NaN/Inf")
        h = self.fc1(x)
        h = self.fc2(h)
        I = self.out(h) + 100
        return I

class ValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = keras.layers.Dense(Config.HIDDEN_UNITS, activation='relu')
        self.fc2 = keras.layers.Dense(Config.HIDDEN_UNITS, activation='relu')
        self.out = keras.layers.Dense(1, activation='linear')

    def call(self, x):
        # x shape: (batch, 2) = [k, z]
        tf.debugging.assert_all_finite(x[:,1], "z has NaN/Inf")
        tf.debugging.assert_all_finite(x[:,0], "k has NaN/Inf")
        h = self.fc1(x)
        h = self.fc2(h)
        I = self.out(h)+ 10
        return I


#-------------------------------------DNNs for risky model-------------------------------------#


class PolicyNet_Risky(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = keras.layers.Dense(Config.HIDDEN_UNITS_RISKY, activation='relu')
        self.fc2 = keras.layers.Dense(Config.HIDDEN_UNITS_RISKY, activation='relu')
        self.out = keras.layers.Dense(2, activation='linear')

    def call(self, x):
        # x shape: (batch, 3) = [k, z, b]
        tf.debugging.assert_all_finite(x[:,0], "k has NaN/Inf")
        tf.debugging.assert_all_finite(x[:,1], "z has NaN/Inf")
        tf.debugging.assert_all_finite(x[:,2], "b has NaN/Inf")

        h = self.fc1(x)
        h = self.fc2(h)
        I = self.out(h) + 100
        return I