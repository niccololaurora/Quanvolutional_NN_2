import math
import random
import tensorflow as tf
import numpy as np
import os
import collections
from tensorflow.keras.datasets import mnist
from qibo import Circuit, gates


def circuit_recipe(filterdim, singleQ, twoQ):
    circuit_recipe_singleQ = []
    random_number = random.randint(1, 2 * filterdim**2)
    for _ in range(random_number):
        random_probability = random.random()
        if random_probability > 0.5:
            random_key_singleQ = random.choice(list(singleQ.keys()))
            random_value_singleQ = singleQ[random_key_singleQ]
            circuit_recipe_singleQ.append(random_value_singleQ)

    nqubits = int(filterdim**2)
    total_possible_combinations = int(
        math.factorial(nqubits) / (2 * math.factorial((nqubits - 2)))
    )
    circuit_recipe_twoQ = []
    for _ in range(total_possible_combinations):
        random_probability = random.random()
        if random_probability > 0.5:
            random_key_twoQ = random.choice(list(twoQ.keys()))
            random_value_twoQ = swoQ[random_key_twoQ]
            circuit_recipe_twoQ.append(random_value_twoQ)

    return circuit_recipe_singleQ, circuit_recipe_twoQ


def couplings(length, filterdim):
    couples = []
    for _ in range(length):
        random_pair = random.sample(range(filterdim**2), 2)
        while random_pair[0] == random_pair[1]:
            random_pair = random.sample(range(filterdim**2), 2)
        couples.append(random_pair)
    return couples


def random_circuit(small_circuit, depth, filterdim, singleQ, twoQ):
    """
    Args: no
    Output: random circuit
    """

    if small_circuit == "yes":
        circuit_recipe_singleQ, circuit_recipe_twoQ = circuit_recipe_small(
            depth, filterdim, singleQ, twoQ
        )
        circuit_recipe = circuit_recipe_singleQ + circuit_recipe_twoQ

    else:
        circuit_recipe_singleQ, circuit_recipe_twoQ = circuit_recipe(
            filterdim, singleQ, twoQ
        )
        circuit_recipe = circuit_recipe_singleQ + circuit_recipe_twoQ

    random.shuffle(circuit_recipe)

    c = Circuit(filterdim**2)
    length = len(circuit_recipe)
    couples = couplings(length, filterdim)

    min_angle = 0
    max_angle = 2 * np.pi

    for x, z in zip(circuit_recipe, couples):
        if x == "GeneralizedfSim":
            matrix = np.array(
                [[1 / 2 + 1j / 2, 1 / 2 - 1j / 2], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2]]
            )
            c.add(gates.GeneralizedfSim(z[0], z[1], unitary=matrix, phi=0))

        if x == "CU3":
            theta = random.uniform(min_angle, max_angle)
            phi = random.uniform(min_angle, max_angle)
            lam = random.uniform(min_angle, max_angle)
            c.add(gates.CU3(z[0], z[1], theta=theta, phi=phi, lam=lam))

        if x == "SWAP":
            c.add(gates.SWAP(z[0], z[1]))

        if x == "CNOT":
            c.add(gates.CNOT(z[0], z[1]))

        if x == "RX":
            theta = random.uniform(min_angle, max_angle)
            c.add(gates.RX(z[0], theta=theta).controlled_by(z[1]))

        if x == "RY":
            theta = random.uniform(min_angle, max_angle)
            c.add(gates.RY(z[0], theta=theta).controlled_by(z[1]))

        if x == "RZ":
            theta = random.uniform(min_angle, max_angle)
            c.add(gates.RZ(z[0], theta=theta).controlled_by(z[1]))

        if x == "U3":
            theta = random.uniform(min_angle, max_angle)
            phi = random.uniform(min_angle, max_angle)
            lam = random.uniform(min_angle, max_angle)
            c.add(gates.U3(z[0], theta=theta, phi=phi, lam=lam).controlled_by(z[1]))

        if x == "S":
            c.add(gates.S(z[0]).controlled_by(z[1]))

        if x == "T":
            c.add(gates.T(z[0]).controlled_by(z[1]))

        if x == "H":
            c.add(gates.H(z[0]).controlled_by(z[1]))

    # add measurement gate for each qubit
    c.add(gates.M(*range(filterdim**2)))

    # print(c.draw())
    return c


def circuit(filterdim):
    c = Circuit(filterdim**2)
    min_angle = 0
    max_angle = 2 * np.pi
    theta1 = random.uniform(min_angle, max_angle)
    theta2 = random.uniform(min_angle, max_angle)
    theta3 = random.uniform(min_angle, max_angle)
    c.add(gates.RX(0, theta=theta1).controlled_by(1))
    c.add(gates.RY(1, theta=theta2).controlled_by(2))
    c.add(gates.RZ(2, theta=theta3).controlled_by(3))
    c.add(gates.CNOT(0, 3))
    c.add(gates.M(*range(filterdim**2)))

    return c


def circuit_recipe_small(depth, filterdim, singleQ, twoQ):
    circuit_recipe_singleQ = []
    random_number = random.randint(1, depth)
    for _ in range(random_number):
        random_probability = random.random()
        if random_probability > 0.5:
            random_key_singleQ = random.choice(list(singleQ.keys()))
            random_value_singleQ = singleQ[random_key_singleQ]
            circuit_recipe_singleQ.append(random_value_singleQ)

    nqubits = int(filterdim**2)
    total_possible_combinations = int(
        math.factorial(nqubits) / (2 * math.factorial((nqubits - 2)))
    )
    circuit_recipe_twoQ = []
    for _ in range(total_possible_combinations):
        random_probability = random.random()
        if random_probability > 0.5:
            random_key_twoQ = random.choice(list(twoQ.keys()))
            random_value_twoQ = twoQ[random_key_twoQ]
            circuit_recipe_twoQ.append(random_value_twoQ)

    return circuit_recipe_singleQ, circuit_recipe_twoQ


def selected_filters(small_circuit, path, nfilter, depth, filterdim, singleQ, twoQ):
    # save the filter used in a file
    if not os.path.isdir(path):
        os.makedirs(path)

    filename = path + "filters.txt"

    filters = []
    for j in range(nfilter):
        quanv_filter, circuit_recipe = random_circuit(
            small_circuit, depth, filterdim, singleQ, twoQ
        )
        filters.append(quanv_filter)

        with open(filename, "a+") as file:
            file.write("\n===========================")
            file.write(f"\nFilter {j+1}")
            file.write(f"\nRecipe: {circuit_recipe}")
            file.write(f"\n{quanv_filter.draw()}")

    return filters


def initial_state(qubits_initialization):
    if qubits_initialization[0] == False:
        initial_state = tf.Variable([1, 0])
    else:
        initial_state = tf.Variable([0, 1])

    for t in range(1, len(qubits_initialization)):
        if qubits_initialization[t] == False:
            state = tf.Variable([1, 0])
            initial_state = tf.tensordot(initial_state, state, axes=0)
        else:
            state = tf.Variable([0, 1])
            initial_state = tf.tensordot(initial_state, state, axes=0)

    return initial_state


def initialize_data(train_size, resize, filt):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if train_size != 0:
        x_train = x_train[0:train_size]
        y_train = y_train[0:train_size]

        x_test[train_size + 1 : (train_size + 1) * 2]
        y_test[train_size + 1 : (train_size + 1) * 2]

    # aggiunge una dimensione alla matrice, ovvero il channel
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    new_size = [resize, resize]
    x_train = tf.image.resize(x_train, new_size)
    x_test = tf.image.resize(x_test, new_size)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # filtro il dataset prendendo solo gli zeri e gli uno
    if filt == "yes":
        mask_train = (y_train == 0) | (y_train == 1)
        mask_test = (y_test == 0) | (y_test == 1)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]

    return x_train, y_train, x_test, y_test


def barplot(x_train, y_train):
    mask_0 = y_train == 0
    mask_1 = y_train == 1
    x_0 = x_train[mask_0]
    x_1 = x_train[mask_1]

    print(len(x_0) + len(x_1))

    digits = {"0": len(x_0), "1": len(x_1)}

    plt.bar(digits.keys(), digits.values(), color="maroon", width=0.4)
    plt.xlabel("Digits")
    plt.title(f"Occurences of 0, 1")
    plt.savefig("barplot.png")


def plot_metrics(history):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="validation")
    ax[0].set_title("QModel Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="validation")
    ax[1].set_title("QModel Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.savefig("fig.png")
    plt.show()


def counter(result, nshots):
    conteggi = []
    for i in range(nshots):
        conteggio = np.count_nonzero(result.samples(binary=True)[i])
        conteggi.append(conteggio)

    mean = int(sum(conteggi) / len(conteggi))


def calculate_qconvolutional_output(image, filterdim, threshold, shots, quanv_filter):
    image_height, image_width = image.shape
    qconvolutional_output_width = image_height - filterdim + 1
    qconvolutional_output_height = image_width - filterdim + 1

    qconvolutional_output = np.zeros(
        (qconvolutional_output_width, qconvolutional_output_height)
    )

    for i in range(qconvolutional_output_width):
        for j in range(qconvolutional_output_height):
            roi = image[i : i + filterdim, j : j + filterdim]
            flattened_roi = tf.reshape(roi, [-1])

            qubits_initialization = flattened_roi > threshold
            print(qubits_initialization)

            init_state = initial_state(qubits_initialization)
            result = quanv_filter(init_state, nshots=shots)

            mean_counts = result.mean()

            qconvolutional_output[i][j] = mean_counts

    qconvolutional_output /= filterdim**2
    return qconvolutional_output
