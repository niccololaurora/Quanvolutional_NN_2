import tensorflow as tf
import numpy as np
from help import initialize_data, barplot, plot_metrics
from quanvolutional_layer import Quanvolutional_Layer, MyModel, CNNBlock


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    nclasses = 2
    filterdim = 2
    nfilter = 5
    train_size = 100
    depth = 10
    resize = 10
    filt = "yes"
    path = "/Users/niccolo/Desktop/mnist/test"

    # load data
    x_train, y_train, x_test, y_test = initialize_data(train_size, resize, filt)

    # my model
    model = MyModel(filterdim, nfilter, depth, nclasses, path)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=2,
        validation_data=(x_test, y_test),
    )

    plot_metrics(history)


if __name__ == "__main__":
    main()
