import tensorflow as tf
import numpy as np
from help import (
    circuit_recipe,
    couplings,
    random_circuit,
    circuit_recipe_small,
    selected_filters,
    counter,
    initial_state,
    barplot,
    plot_metrics,
    circuit,
    calculate_qconvolutional_output,
)


"""
This class inherits from tf.Keras.Model, hence I will access to:
model.compile, model.fit, model.evaluate
"""


class MyModel(tf.keras.Model):
    def __init__(
        self, filterdim, nfilter, depth, nclasses, x_train, y_train, path=None
    ):
        super(MyModel, self).__init__()
        self.Quanv_layer = Quanvolutional_Layer(
            filterdim=filterdim, nfilter=nfilter, depth=depth, path=path
        )
        self.CNNBlock = CNNBlock(nclasses=nclasses)
        self.x_train = x_train
        self.y_train = y_train

    def call(self, x):
        output = self.CNNBlock(self.Quanv_layer(x))
        return output


"""
This is my Quanvolutional Layer.
"""


class Quanvolutional_Layer(tf.keras.layers.Layer):
    def __init__(self, filterdim=2, nfilter=5, depth=5, resize=10, path=None):
        super(Quanvolutional_Layer, self).__init__()
        self.filterdim = filterdim
        self.nfilter = nfilter
        self.depth = depth
        self.small_circuit = "yes"
        self.width = 0
        self.height = 0
        self.shots = 1000
        self.threshold = 0.5
        self.path = path
        self.singleQ = {
            "X-rotation": "RX",
            "Y-rotation": "RY",
            "Z-rotation": "RZ",
        }
        self.twoQ = {
            "Cnot": "CNOT",
        }

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        # values = initializer(shape=(1, 3))
        self.params = self.add_weight(
            shape=(3, 1),
            initializer=initializer,
            trainable=True,
        )
        self.random_circuit = circuit(self.filterdim)
        print(f"Params: {self.params}")
        self.random_circuit.set_parameters(self.params)

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        # new_image = []

        output = self.quanvolutional_filter(inputs, self.random_circuit)
        # new_image.append(output)

        # final_output = tf.stack(new_image, axis=-1)
        return output

    def quanvolutional_filter(self, image, quanv_filter):
        """
        Args: an image and a random circuit
        Output: a matrix. If the image is 6x6 and filter 3x3 the output
        matrix will be 4x4
        """
        # sopprimo la dimensione del channel
        # print(f"Quanv filter {image.shape}")

        image = image[0, :, :, 0]

        # Boundaries of sliding window
        image_heigth, image_width = image.shape
        qconvolutional_output_width = image_width - self.filterdim + 1
        qconvolutional_output_heigth = image_heigth - self.filterdim + 1
        # self.width = qconvolutional_output_width
        # self.height = qconvolutional_output_heigth
        print("Before calculate_qconvolutional_output")
        print(self.random_circuit.draw())
        qconvolutional_output = calculate_qconvolutional_output(
            image,
            self.filterdim,
            self.threshold,
            self.shots,
            quanv_filter,
        )

        return qconvolutional_output


"""
Blocco della CNN.
"""


class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, nclasses):
        super(CNNBlock, self).__init__()
        self.nclasses = nclasses

    def call(self, inputs):
        print(f"inputs shape CNNBLOCk {inputs.shape}")
        expanded_inputs = inputs[tf.newaxis, :, :, tf.newaxis]
        print(f"inputs shape CNNBLOCk {expanded_inputs.shape}")
        x = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", input_shape=(9, 9))(
            expanded_inputs
        )
        x = tf.keras.layers.AveragePooling2D()(x)
        x = tf.keras.layers.Conv2D(128, (2, 2), activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        output = tf.keras.layers.Dense(self.nclasses, activation="softmax")(x)

        return output
