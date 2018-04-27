#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jul  5 21:37:01 2017

@author: chavdar
"""


import keras
from keras.layers import Activation
from keras.layers import Dense
# from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
# from keras.layers import MaxPooling2D
from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn import neural_network


class QNN:
    def __init__(self, state_size, action_size, learn_rate):
        self.state_size = state_size
        self.action_size = action_size

        self.nn = neural_network.MLPRegressor(hidden_layer_sizes=(20,),
                                              activation='relu', solver='sgd',
                                              learning_rate_init=learn_rate,
                                              max_iter=1)

    def train(self, dataset, iterations):
        X = []
        Y = []

        for data in dataset:
            X.append(data["input"])
            Y.append(data["output"])

        self.nn.fit(X, Y)

    def forwardProp(self, state):
        try:
            return [self.nn.predict(state)]
        except:  # TODO: Exception type reqiured
            return [[0]*self.action_size]


class QNN_keras:
    def __init__(self, state_size, action_size, learn_rate, hidden):
        self.state_size = state_size
        self.action_size = action_size

        self.nn = Sequential()
        for l in hidden:
            self.nn.add(Dense(l, activation='relu', input_dim=state_size))

        self.nn.add(Dense(self.action_size, activation='linear'))

        # self.model.add(Activation('softmax'))
        # opt = keras.optimizers.RMSprop(lr=learn_rate)

        # opt = keras.optimizers.SGD(lr=learn_rate, momentum=0.5)
        opt = keras.optimizers.Adam(lr=learn_rate)

        self.nn.compile(loss=keras.losses.mean_squared_error, optimizer=opt)

    def train(self, dataset, iterations):
        X = []
        Y = []

        for data in dataset:
            X.append(data["input"])
            Y.append(data["output"])

        X = np.array(X)
        Y = np.array(Y)

        # print("training")
        # print(X, Y)

        # for i in range(iterations):
        #    print(self.nn.train_on_batch(X, Y))

        callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                 min_delta=0.005,
                                                 patience=100, verbose=0,
                                                 mode='auto')

        self.nn.fit(X, Y, batch_size=X.shape[0], epochs=iterations,
                    verbose=0, callbacks=[callback], validation_split=0.0,
                    validation_data=None, shuffle=True, class_weight=None,
                    sample_weight=None, initial_epoch=0)

    def train2(self, X, Y, batchSize=32, iterations=1):
        callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                 min_delta=0.005,
                                                 patience=2,
                                                 verbose=0,
                                                 mode='auto')

        self.nn.fit(X, Y, batch_size=batchSize, epochs=iterations,
                    verbose=2, callbacks=[callback], validation_split=0.0,
                    validation_data=None, shuffle=True, class_weight=None,
                    sample_weight=None, initial_epoch=0)

    def forwardProp(self, state):
        np_state = np.array(state)
        np_state = np_state.reshape((1, self.state_size))
        # print(state, np_state)
        return self.nn.predict(np_state, batch_size=1)

    def forwardProp2(self, np_dataX, batch_size):
        # print(state, np_state)
        return self.nn.predict(np_dataX, batch_size=batch_size)


def getCNNModel(state_shape, action_size, params):
    model = Sequential()
    initial = 1
    params_nn = params["architecture"]

    for l in params_nn:
        if l[0] == "conv":
            if initial:
                model.add(Conv2D(l[1], (l[2], l[2]), strides=(l[3], l[3]),
                                 padding="same", input_shape=state_shape))
                model.add(Activation('relu'))
                initial = 0
            else:
                model.add(Conv2D(l[1], (l[2], l[2]), strides=(l[3], l[3]),
                                 padding="same"))
                model.add(Activation('relu'))
        if l[0] == "gap":
            model.add(GlobalAveragePooling2D())
        elif l[0] == "flatten":
            model.add(Flatten())
        if l[0] == "fc":
            model.add(Dense(l[1], activation='relu'))

    model.add(Dense(action_size, activation='linear'))

    # Optimiser
    learn_rate = params["learn_rate"]
    if params["optimizer"] == "RMSprop":
        opt = keras.optimizers.RMSprop(lr=learn_rate)
    elif params["optimizer"] == "Adam":
        opt = keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss=keras.losses.mean_squared_error, optimizer=opt)

    return model


class QCNN_keras:
    def __init__(self, state_shape, action_size, learn_rate, hidden):
        self.state_shape = state_shape
        self.action_size = action_size
        self.nn = getCNNModel(state_shape, action_size, hidden)

        """
        self.nn = Sequential()
        #for i in hidden:
        #    self.nn.add(Dense(i, activation='relu', input_dim=state_size))
        self.nn.add(Conv2D(30 ,(6,6),strides=(2,2),padding="same",
                 input_shape=state_shape))
        self.nn.add(Activation('relu'))
        #self.nn.add(MaxPooling2D(pool_size=(2, 2)))
        self.nn.add(Conv2D(40, (4, 4),strides=(2,2),padding="same"))
        self.nn.add(Activation('relu'))
        self.nn.add(Conv2D(60, (3, 3),strides=(1,1),padding="same"))
        self.nn.add(Activation('relu'))
        #self.nn.add(Conv2D(10, (3, 3),strides=(1,1)))
        #self.nn.add(Activation('relu'))
        self.nn.add(GlobalAveragePooling2D())
        #self.nn.add(Flatten())
        self.nn.add(Dense(516, activation='relu'))
        self.nn.add(Dense(self.action_size, activation='linear'))

        #self.model.add(Activation('softmax'))

        opt = keras.optimizers.RMSprop(lr=learn_rate)
        ### debug
        #self.nn = keras.models.load_model("m10_catch23.9")
        #############
        #opt = keras.optimizers.SGD(lr=learn_rate, momentum=0.5)
        #opt = keras.optimizers.Adam(lr=learn_rate)
        #opt = keras.optimizers.Adagrad()
        self.nn.compile(loss=keras.losses.mean_squared_error,
              optimizer=opt)
        """

    def train(self, dataset, iterations):
        X = []
        Y = []
        for i in dataset:
            X.append(i["input"])
            Y.append(i["output"])

        X = np.array(X)
        Y = np.array(Y)

        # print("training")
        # print(x,y)
        # for i in range(iterations):
        #    print(self.nn.train_on_batch(x, y))

        # callback = keras.callbacks.EarlyStopping(monitor='loss',
        #                                          min_delta=0.005,
        #                                          patience=2,
        #                                          verbose=0,
        #                                          mode='auto')

        self.nn.fit(X, Y, batch_size=X.shape[0], epochs=iterations, verbose=0,
                    callbacks=[], validation_split=0.0, validation_data=None,
                    shuffle=True, class_weight=None, sample_weight=None,
                    initial_epoch=0)

    def train2(self, X, Y, batchSize=32, iterations=1, verbose=0, vsplit=0.0):
        # callback = keras.callbacks.EarlyStopping(monitor='loss',
        #                                          min_delta=0.005,
        #                                          patience=2,
        #                                          verbose=0,
        #                                          mode='auto')

        self.nn.fit(X, Y, batch_size=batchSize, epochs=iterations,
                    verbose=verbose, callbacks=[], validation_split=vsplit,
                    validation_data=None, shuffle=False)

    def forwardProp(self, state):
        # np_state = np.array(state)
        # np_state = np_state.reshape((1,self.state_shape))
        # print(state,np_state)
        ss = self.state_shape[:]
        ss.insert(0, 1)
        return self.nn.predict(np.reshape(state, ss), batch_size=1)

    def forwardProp2(self, np_dataX, batch_size):

        # print(state,np_state)
        return self.nn.predict(np_dataX, batch_size=batch_size)
