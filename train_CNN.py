import numpy as np
import cv2
import glob
import feature_extraction as fe
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import DataSet_Preparation as pre

# Fixing seed generator as instructed to reproduce results
seed = 7
np.random.seed(seed)

num_digits_classes=10
dropout=0.2
# CNN Model
def CNN_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_digits_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_arabic_CNN():
    X_train,Y_train= pre.prepare_train()
    X_train=np.reshape(X_train,(len(X_train), 28, 28, 1))
    Y_train=np_utils.to_categorical(Y_train)

    model = CNN_model()
    model.fit(X_train, Y_train,epochs=10)
    return model

def train_english_CNN():
    X_train,Y_train= pre.prepare_english_train()
    X_train=np.reshape(X_train,(len(X_train), 28, 28, 1))
    Y_train=np_utils.to_categorical(Y_train)

    model = CNN_model()
    model.fit(X_train, Y_train,epochs=10)
    return model
