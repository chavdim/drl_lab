#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:37:01 2017

@author: chavdar
"""
from sklearn import neural_network
from keras.models import load_model
class QNN:
    def __init__(self,state_size,action_size,learn_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.nn = neural_network.MLPRegressor(hidden_layer_sizes=(20,),
                                              activation='relu', solver='sgd',
                                              learning_rate_init =learn_rate,
                                              max_iter=1)
    def train(self,dataSet,iterations):
        x = []
        y = []
        for i in dataSet:
            x.append(i["input"])
            y.append(i["output"])
        self.nn.fit(x,y)
    def forwardProp(self,state):
        try:
            return [self.nn.predict(state)]
        except:
            return [[0]*self.action_size]
       
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
class QNN_keras:
    def __init__(self,state_size,action_size,learn_rate,hidden):
        self.state_size = state_size
        self.action_size = action_size
        self.nn = Sequential()
        for i in hidden:
            self.nn.add(Dense(i, activation='relu', input_dim=state_size))
        self.nn.add(Dense(self.action_size, activation='linear'))

        #self.model.add(Activation('softmax'))
        #opt = keras.optimizers.RMSprop(lr=learn_rate)

        #opt = keras.optimizers.SGD(lr=learn_rate, momentum=0.5)
        opt = keras.optimizers.Adam(lr=learn_rate)
        self.nn.compile(loss=keras.losses.mean_squared_error,
              optimizer=opt)
               
    def train(self,dataSet,iterations):
        x = []
        y = []
        for i in dataSet:
            x.append(i["input"])
            y.append(i["output"])

        x = np.array(x)
        y = np.array(y)
        #print("training")
        #print(x,y)
        #for i in range(iterations):
            #print(self.nn.train_on_batch(x, y))
        callbk = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=100, verbose=0, mode='auto')

        self.nn.fit(x, y, batch_size=x.shape[0], epochs=iterations, verbose=0, callbacks=[callbk], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    def train2(self,X,y,batchSize=32,iterations=1):
        callbk = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=2, verbose=0, mode='auto')

        self.nn.fit(X, y, batch_size=batchSize, epochs=iterations, verbose=2, callbacks=[callbk], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
        
    def forwardProp(self,state):
        np_state = np.array(state)
        np_state = np_state.reshape((1,self.state_size))
        #print(state,np_state)
        return self.nn.predict(np_state,batch_size=1)
        
    def forwardProp2(self,np_dataX,batch_size):

        #print(state,np_state)
        return self.nn.predict(np_dataX,batch_size=batch_size)
def getCNNModel(state_shape,action_size,params):
    model = Sequential()
    initial = 1
    params_nn = params["architecture"]
    for i in params_nn:
        if i[0]=="conv":
            if initial:
                model.add(Conv2D(i[1] ,(i[2],i[2]),strides=(i[3],i[3]),padding="same",
                                 input_shape=state_shape))
                model.add(Activation('relu'))
                initial=0
            else:
                model.add(Conv2D(i[1] ,(i[2],i[2]),strides=(i[3],i[3]),padding="same"))
                model.add(Activation('relu'))
        if i[0]=="gap":
            model.add(GlobalAveragePooling2D())
        elif i[0]=="flatten":
            model.add(Flatten())
        if i[0]=="fc":
            model.add(Dense(i[1], activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    ##### Optimiser
    learn_rate = params["learn_rate"]
    if params["optimizer"] == "RMSprop":
        opt = keras.optimizers.RMSprop(lr=learn_rate)
    elif params["optimizer"] == "Adam":
        opt = keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=opt)
    return model
    
class QCNN_keras:
    def __init__(self,state_shape,action_size,learn_rate,hidden):
        self.state_shape = state_shape
        self.action_size = action_size
        self.nn = getCNNModel(state_shape,action_size,hidden)
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
               
        
    def train(self,dataSet,iterations):
        x = []
        y = []
        for i in dataSet:
            x.append(i["input"])
            y.append(i["output"])

        x = np.array(x)
        y = np.array(y)
        #print("training")
        #print(x,y)
        #for i in range(iterations):
            #print(self.nn.train_on_batch(x, y))
        callbk = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=2, verbose=0, mode='auto')

        self.nn.fit(x, y, batch_size=x.shape[0], epochs=iterations, verbose=0, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    def train2(self,X,y,batchSize=32,iterations=1,verbose=0,vsplit=0.0):
        #callbk = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=2, verbose=0, mode='auto')

        self.nn.fit(X, y, batch_size=batchSize, epochs=iterations, verbose=verbose, callbacks=[], validation_split=vsplit, validation_data=None, shuffle=False)
        
    def forwardProp(self,state):
        #np_state = np.array(state)
        #np_state = np_state.reshape((1,self.state_shape))
        #print(state,np_state)
        ss = self.state_shape[:]
        ss.insert(0,1)
        return self.nn.predict(np.reshape(state,ss),batch_size=1)
        
    def forwardProp2(self,np_dataX,batch_size):

        #print(state,np_state)
        return self.nn.predict(np_dataX,batch_size=batch_size)
        
        
        
        
        
        