import tensorflow as tf
import numpy as np
import time
import random
import math
import os
import matplotlib.pyplot as plt

from qibo import Circuit, gates
from tensorflow.keras.datasets import mnist




'''
0. Data: initialize_data ---> Scarica i dati, sceglie quante immagini voglio, ne fa il resizing e normalizing, e filtra se lo voglio
1. Random circuit creator: random_circuit
2. Quanvolutional filter: quanvolutional_filter --> fa la convoluzione di una immagine ---> devo far in modo che faccia la convoluzione
per gli n filtri
3. Quanvolutional layer: applica il quanvolutional_filter molte volte ---> devo far in modo che il suo output sia (nimages, width, height, nfilters)
4. Classical model: classical_model ----> devo aggiungere inputs pari a (nimages, width, height, nfilters)
5. Training loop: training_loop
6. Plot:

'''




class Quanvolutional_NN():

    def __init__(self, filterdim = 3, train_size = 1000, circuit_small = "yes"):
        self.singleQ = {
            'X-rotation' : 'RX',
            'Y-rotation' : 'RY',
            'Z-rotation' : 'RZ',
            'Generic Unitary' : 'U3',
            'Phase gate' : 'S',
            'T-gate' : 'T',
            'Hadamard' : 'H'
            }
        self.twoQ = {
            'Cnot' : 'CNOT',
            'Swap' : 'SWAP',
            'sqrtSwap' : 'GeneralizedfSim',
            'ControlledU' : 'CU3'
            }

        
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0

        self.train_size = train_size
        self.threshold = 0.5 
        self.classes = 2
        self.depth = 10 # nel paper 2*filterdim**2, quindi per filterdim=2 ---> 16
        
        self.filter = "yes" #se filtrare o meno i dati per avere solo due digits
        self.shots = 20

        self.nfilter = 5
        self.width = 0
        self.heigth = 0
        self.epochs = 20
        self.validationsize = 0.2
        self.batch_size = 16
        self.filterdim = filterdim
        self.small_circuit = circuit_small
        self.name = "Test_" + str(self.train_size) + "_" + str(self.epochs)
        self.path = "/Users/niccolo/Desktop/mnist/results/" + str(filterdim) + "x" + str(filterdim) 
            + "_filter" + "/Filters_" + str(self.nfilter) + "/" + self.name + "/"

    def initialize_data(self, dimension):

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if self.train_size != 0:
            x_train = x_train[0:self.train_size]
            y_train = y_train[0:self.train_size]

            x_test[self.train_size + 1: (self.train_size + 1)*2]
            y_test[self.train_size + 1: (self.train_size + 1)*2]


        # aggiunge una dimensione alla matrice, ovvero il channel 
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
  
        new_size = [dimension, dimension]
        x_train = tf.image.resize(x_train, new_size)
        x_test = tf.image.resize(x_test, new_size)

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # filtro il dataset prendendo solo gli zeri e gli uno
        if self.filter == "yes":
            mask_train = ((y_train == 0) | (y_train == 1))
            mask_test = ((y_test == 0) | (y_test == 1))
            x_train = x_train[mask_train]
            y_train = y_train[mask_train]
            x_test = x_test[mask_test]
            y_test = y_test[mask_test]


        self.ntrain = len(x_train)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        print(f"Initialize data: {self.x_train.shape}")

    def counter(self, result, nshots):

        conteggi = []
        for i in range(nshots):
            conteggio = np.count_nonzero(result.samples(binary=True)[i])
            conteggi.append(conteggio)

        mean = int(sum(conteggi) / len(conteggi))

        return mean

    def initial_state(self, qubits_initialization):
        if qubits_initialization[0] == 0:
            initial_state = tf.Variable([1,0])
        else:
            initial_state = tf.Variable([0,1])

        for t in range(1, len(qubits_initialization)):
            if qubits_initialization[t] == 0:
                state = tf.Variable([1,0])
                initial_state = tf.tensordot(initial_state, state, axes=0)
            else:
                state = tf.Variable([0,1])
                initial_state = tf.tensordot(initial_state, state, axes=0)

        return initial_state

    def random_circuit(self):
        '''
        Args: no
        Output: random circuit 
        '''

        if self.small_circuit == "yes":
            circuit_recipe_singleQ, circuit_recipe_twoQ = self.circuit_recipe_small()
            circuit_recipe = circuit_recipe_singleQ + circuit_recipe_twoQ

        else:
            circuit_recipe_singleQ, circuit_recipe_twoQ = self.circuit_recipe()
            circuit_recipe = circuit_recipe_singleQ + circuit_recipe_twoQ

        random.shuffle(circuit_recipe)

        c = Circuit(self.filterdim ** 2)
        length = len(circuit_recipe)
        couples = self.couplings(length)

        min_angle = 0
        max_angle = 2 * np.pi

        for x, z in zip(circuit_recipe, couples):

            if x == 'GeneralizedfSim':
                matrix = np.array([[1/2+1j/2, 1/2-1j/2], 
                                    [1/2-1j/2, 1/2+1j/2]])
                c.add(gates.GeneralizedfSim(z[0], z[1], unitary = matrix, phi = 0))
            
            if x == 'CU3':
                theta = random.uniform(min_angle, max_angle)
                phi = random.uniform(min_angle, max_angle)
                lam = random.uniform(min_angle, max_angle)
                c.add(gates.CU3(z[0], z[1], theta = theta, phi = phi, lam = lam))

            if x == 'SWAP':
                c.add(gates.SWAP(z[0], z[1]))

            if x == 'CNOT':
                c.add(gates.CNOT(z[0], z[1]))

            if x == 'RX':
                theta = random.uniform(min_angle, max_angle)
                c.add(gates.RX(z[0], theta = theta).controlled_by(z[1]))

            if x == 'RY':
                theta = random.uniform(min_angle, max_angle)
                c.add(gates.RY(z[0], theta = theta).controlled_by(z[1]))

            if x == 'RZ':
                theta = random.uniform(min_angle, max_angle)
                c.add(gates.RZ(z[0], theta = theta).controlled_by(z[1]))
            
            if x == 'U3':
                theta = random.uniform(min_angle, max_angle)
                phi = random.uniform(min_angle, max_angle)
                lam = random.uniform(min_angle, max_angle)
                c.add(gates.U3(z[0], theta = theta, phi = phi, lam = lam).controlled_by(z[1]))

            if x == 'S':
                c.add(gates.S(z[0]).controlled_by(z[1]))

            if x == 'T':
                c.add(gates.T(z[0]).controlled_by(z[1]))

            if x == 'H':
                c.add(gates.H(z[0]).controlled_by(z[1]))

        # add measurement gate for each qubit 
        c.add(gates.M(*range(self.filterdim ** 2)))

        # print(c.draw())
        return c, circuit_recipe

    def circuit_recipe_small(self):

        circuit_recipe_singleQ = []
        random_number = random.randint(1, self.depth)
        for _ in range(random_number):
            random_probability = random.random()
            if random_probability > 0.5:
                random_key_singleQ = random.choice(list(self.singleQ.keys()))
                random_value_singleQ = self.singleQ[random_key_singleQ]
                circuit_recipe_singleQ.append(random_value_singleQ)

        nqubits = int(self.filterdim ** 2)
        total_possible_combinations = int(math.factorial(nqubits) / (2 * math.factorial((nqubits - 2))))
        circuit_recipe_twoQ = []
        for _ in range(total_possible_combinations):
            random_probability = random.random()
            if random_probability > 0.5:
                random_key_twoQ = random.choice(list(self.twoQ.keys()))
                random_value_twoQ = self.twoQ[random_key_twoQ]
                circuit_recipe_twoQ.append(random_value_twoQ)

        return circuit_recipe_singleQ, circuit_recipe_twoQ

    def circuit_recipe(self):
        circuit_recipe_singleQ = []
        random_number = random.randint(1, 2*self.filterdim**2)
        for _ in range(random_number):
            random_probability = random.random()
            if random_probability > 0.5:
                random_key_singleQ = random.choice(list(self.singleQ.keys()))
                random_value_singleQ = self.singleQ[random_key_singleQ]
                circuit_recipe_singleQ.append(random_value_singleQ)

        nqubits = int(self.filterdim ** 2)
        total_possible_combinations = int(math.factorial(nqubits) / (2 * math.factorial((nqubits - 2))))
        circuit_recipe_twoQ = []
        for _ in range(total_possible_combinations):
            random_probability = random.random()
            if random_probability > 0.5:
                random_key_twoQ = random.choice(list(self.twoQ.keys()))
                random_value_twoQ = self.twoQ[random_key_twoQ]
                circuit_recipe_twoQ.append(random_value_twoQ)

        return circuit_recipe_singleQ, circuit_recipe_twoQ

    def couplings(self, length):
        couples = []
        for _ in range(length):
            random_pair = random.sample(range(self.filterdim ** 2), 2)
            while random_pair[0] == random_pair[1]:
                random_pair = random.sample(range(self.filterdim ** 2), 2)
            couples.append(random_pair)
        return couples

    def selected_filters(self):
        # save the filter used in a file
        if not os.path.isdir(self.path):
            os.makedirs(self.path) 

        filename = self.path + "filters.txt"

        filters = []
        for j in range(self.nfilter):
            quanv_filter, circuit_recipe = self.random_circuit()
            filters.append(quanv_filter)

            with open(filename, "a+") as file:
                file.write("\n===========================")
                file.write(f"\nFilter {j+1}")
                file.write(f"\nRecipe: {circuit_recipe}")
                file.write(f"\n{quanv_filter.draw()}")
        
        return filters

    def quanvolutional_filter(self, image, quanv_filter):
        '''
        Args: an image and a random circuit
        Output: a matrix. If the image is 6x6 and filter 3x3 the output
        matrix will be 4x4
        '''
        # sopprimo la dimensione del channel
        image = image[:, :, 0]
    
        # Boundaries of sliding window
        image_heigth, image_width = image.shape
        qconvolutional_output_width = image_width - self.filterdim + 1
        qconvolutional_output_heigth = image_heigth - self.filterdim + 1
        self.width = qconvolutional_output_width
        self.height = qconvolutional_output_heigth

        # matrice qconvolutional_output_width x qconvolutional_output_heigth 
        qconvolutional_output = np.zeros((qconvolutional_output_width, qconvolutional_output_heigth))
        
        for i in range(qconvolutional_output_heigth):
            qubits_initialization = np.array([0] * self.filterdim **2) 
            
            for j in range(qconvolutional_output_heigth):
                # Region of interest (ROI) of the image
                roi = image[i:i+self.filterdim, j:j+self.filterdim]
                flattened_roi = tf.reshape(roi, [-1])
                
                for k in range(len(flattened_roi)):
                    if flattened_roi[k] > self.threshold:
                        qubits_initialization[k] = 1

                # calcolo lo stato iniziale
                initial_state = self.initial_state(qubits_initialization)
                
                # calcolo lo stato finale e lo misuro N volte
                result = quanv_filter(initial_state, nshots=self.shots)

                # conto il numero di volte in cui nello stato finale c'è |1> per le N misure e faccio media
                nshots = self.shots
                mean_counts = self.counter(result, nshots)

                # Metto il valore medio nella matrice
                qconvolutional_output[i][j] = mean_counts

                # Normalizzo la matrice
                qconvolutional_output = qconvolutional_output / self.filterdim**2

        return qconvolutional_output

    def quanvolutional_layer(self):

        filters = self.selected_filters()

        new_x_train = []
        for i, x in enumerate(self.x_train):

            image = []
            for j in range(self.nfilter):
                # creo un circuito random
                output = self.quanvolutional_filter(x, filters[j])
                image.append(output)

            new_x = tf.stack(image, axis=-1)
            new_x_train.append(new_x)

            if i % 20 == 0:
                print(f"Quantum encoded image: {i+1}")

        x_train = tf.stack(new_x_train, axis=0)

        self.x_train = x_train
         
    def classical_model(self):
        print("Building the model")
        model = tf.keras.models.Sequential()
        #model.add(tf.keras.layers.InputLayer(input_shape=(self.ntrain, self.width, self.heigth, self.nfilter)))
        model.add(tf.keras.layers.Conv2D(128, (2,2), activation='relu'))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        model.add(tf.keras.layers.Conv2D(64, (2,2), activation='relu'))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(30, activation='relu'))

        if self.classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='classifier'))

        else:
            model.add(tf.keras.layers.Dense(self.classes, activation='softmax', name='classifier'))

        print("Model successfully builded")
        return model

    def training_loop(self):

        model = self.classical_model()

        if self.classes == 2: 
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        else: 
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("Shape attesa: (N, 8, 8, 5)")
        print(f"Training Loop {self.x_train.shape}")
        history = model.fit(self.x_train, self.y_train, epochs = self.epochs, batch_size = self.batch_size, 
            validation_split = self.validationsize)

        return history

    def barplot(self):

        mask_0 = ((self.y_train == 0))
        mask_1 = ((self.y_train == 1))
        x_0 = self.x_train[mask_0]
        x_1 = self.x_train[mask_1]

        print(len(x_0) + len(x_1))

        digits = {'0': len(x_0),
                    '1': len(x_1)}

        plt.bar(digits.keys(), digits.values(), color ='maroon', width = 0.4)
        plt.xlabel("Digits")
        plt.title(f"Occurences of 0, 1")
        plt.savefig("barplot.png")

    def plot_metrics(self, history):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

        ax[0].plot(history.history['loss'], label='train')
        ax[0].plot(history.history['val_loss'], label='validation')
        ax[0].set_title('QModel Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].legend()
        

        ax[1].plot(history.history['accuracy'], label='train')
        ax[1].plot(history.history['val_accuracy'], label='validation')
        ax[1].set_title('QModel Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].legend()

        plt.savefig("fig.png")
        plt.show()