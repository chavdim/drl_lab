import keras
from keras.layers import Activation
from keras.layers import Dense
# from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
# from keras.layers import MaxPooling2D
from keras.models import Sequential
import numpy as np


class QCNN:
    def __init__(self, input_shape, output_size, hparams):
        # Build Network
        model = Sequential()
        initial = True
        for l in hparams["layers"]:
            if l[0] == "conv":
                if initial:
                    model.add(Conv2D(l[1], (l[2], l[2]), strides=(l[3], l[3]),
                                     padding="same", input_shape=input_shape))
                    initial = False
                else:
                    model.add(Conv2D(l[1], (l[2], l[2]), strides=(l[3], l[3]),
                                     padding="same"))
                model.add(Activation('relu'))
            elif l[0] == "gap":
                model.add(GlobalAveragePooling2D())
            elif l[0] == "flatten":
                model.add(Flatten())
            elif l[0] == "fc":
                model.add(Dense(l[1], activation='relu'))
        model.add(Dense(output_size, activation='linear'))

        # Set Optimiser
        learn_rate = hparams["learn_rate"]
        if hparams["optimizer"] == "RMSprop":
            opt = keras.optimizers.RMSprop(lr=learn_rate)
        elif hparams["optimizer"] == "Adam":
            opt = keras.optimizers.Adam(lr=learn_rate)

        model.compile(loss=keras.losses.mean_squared_error, optimizer=opt)
        self.nn = model

        # overwrite model with saved_model
        saved_model = hparams['saved_model']
        if saved_model is not None:
            self.load_model(saved_model)
            print('model loaded.')

    def train(self, X, Y, batch_size, epochs=1,
              verbose=0, vsplit=0.0, shuffle=False):
        self.nn.fit(X, Y, batch_size=batch_size, epochs=epochs,
                    verbose=verbose, callbacks=[], validation_split=vsplit,
                    validation_data=None, shuffle=False)

    def forward_prop(self, np_dataX, batch_size):
        return self.nn.predict(np_dataX, batch_size=batch_size)

    def load_model(self, name):
        self.nn = keras.models.load_model(name)


def dataset2XY(dataset):
    X, Y = [], []
    for data in dataset:
        X.append(data["input"])
        Y.append(data["output"])
    X, Y = np.array(X), np.array(Y)
    return X, Y


def state2data(state):
    # add batch size
    ss = list(state.shape)
    ss.insert(0, 1)
    state = np.reshape(state, ss)
    return state
