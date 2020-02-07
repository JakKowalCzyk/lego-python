from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class CNN:
    @staticmethod
    def build():

        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size = (1,1),input_shape=(64,64,1),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters = 64,kernel_size = (3,3),activation= 'relu',padding='same'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters = 128,kernel_size = (3,3),activation= 'relu',padding='same'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters = 128,kernel_size = (3,3),activation= 'relu',padding='same'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(150))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2,activation = 'softmax'))
        model.summary()
        return model
