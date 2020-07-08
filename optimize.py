# Data wrangling and visualization imports
import numpy as np
import pandas as pd
# Keras Imports
import tensorflow as tf
import keras
from keras import backend as K
# CNN and MLP architecture
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    Flatten,
    BatchNormalization
)
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import RandomNormal
# Keras Callbacks
from keras.callbacks import TensorBoard
# Image Preprocessing
from PIL import Image
from keras.preprocessing.image import img_to_array
# Optimizing Hyperparameters
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
# Import generator
import util


def data():
    # get the data
    df = pd.read_csv('Fire-Detection-Image-Dataset/fires.csv')
    selector = np.random.rand(len(df)) < 0.75
    df_train, df_test = df[selector], df[~selector]
    return df_train, df_test


def optimize_model():
    """Returns a dictionary with the results of the best model."""
    # define known parameters
    layer_size = 4
    # Instaniate model
    model = Sequential()
    # Instantiate TensorBoard to visualize model performance
    tensorboard = TensorBoard(log_dir='./Graph')
    # Add 3 CNN layers
    model.add(Conv2D(layer_size, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(1024, 1024, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(layer_size,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(layer_size,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the data
    model.add(Flatten())
    # Add 1 MLP Layer
    model.add(Dense(1))
    # Compile Model
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=
                      {{choice(
                          [keras.optimizers.Adadelta(),
                          'sgd', 'adam', 'rmsprop']   
                           )}},
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    # Train Model
    batch_size = {{choice([10, 20, 40])}}
    result = model.fit_generator(generator=util.data_gen(df_train,
                        batch_size=batch_size),
                        steps_per_epoch=len(df_train['label']) // batch_size,
                        epochs={{choice([3, 5, 7])}},
                        validation_data=util.data_gen(df_test, batch_size=batch_size),
                        validation_steps=len(df_test['label']) // batch_size, 
                        callbacks=[tensorboard])
    # Get Optimized results
    # get the highest validation metrics of the training epochs
    val_loss = np.amax(result.history['val_loss'])
    print(f'Best validation acc of epoch: {(1-val_loss)}')
    return {
        'loss': val_loss,
        'accuracy': 1-val_loss,
        'status': STATUS_OK,
        'model': model
    }


if __name__ == "__main__":
    # find the best model!
    best_run, best_model = optim.minimize(model=optimize_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials())
    print("Best performing model chosen hyper-parameters:")
    print(best_run)