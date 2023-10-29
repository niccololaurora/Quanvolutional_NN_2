import tensorflow as tf
import numpy as np
from help import circuit_recipe, couplings, random_circuit, circuit_recipe_small, selected_filters, counter, initial_state, barplot, plot_metrics



'''
This class inherits from tf.Keras.Model, hence I will access to:
model.compile, model.fit, model.evaluate
'''
class MyModel(tf.keras.Model):
    def __init__(self, filterdim, nfilter, depth, nclasses, path = None):
        super(MyModel, self).__init__()
        self.Quanv_layer = Quanvolutional_Layer(filterdim = filterdim, nfilter = nfilter, depth= depth, path = path)
        self.CNNBlock = CNNBlock(nclasses = nclasses)

    def call(self, x):
        output = self.CNNBlock(self.Quanv_layer(x))
        return output

'''
This is my Quanvolutional Layer.
'''
class Quanvolutional_Layer(tf.keras.layers.Layer):
    def __init__(self, filterdim = 2, nfilter = 5, depth = 10, resize = 10, path = None):
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


    # Defines the computation from inputs to outputs
    def call(self, inputs):
        
        print(f"Call {inputs.shape}")
        filters = selected_filters(self.small_circuit, self.path, self.nfilter, self.depth, self.filterdim, self.singleQ, self.twoQ)
        new_image = []
        #print(f"Shape: {image.shape}")
        for j in range(self.nfilter):
            print(f"for {image.inputs}")
            output = self.quanvolutional_filter(inputs, filters[j])
            new_image.append(output)
    
        final_output = tf.stack(new_image, axis=-1)
        return final_output


    def quanvolutional_filter(self, image, quanv_filter):
        '''
        Args: an image and a random circuit
        Output: a matrix. If the image is 6x6 and filter 3x3 the output
        matrix will be 4x4
        '''
        # sopprimo la dimensione del channel
        print(f"Quanv filter {image.shape}")

        image = image[0, :, :, 0]
    
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
                mean_counts = counter(result, nshots)

                # Metto il valore medio nella matrice
                qconvolutional_output[i][j] = mean_counts

                # Normalizzo la matrice
                qconvolutional_output = qconvolutional_output / self.filterdim**2

        return qconvolutional_output

'''
Blocco della CNN.
'''
class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, nclasses):
        super(CNNBlock, self).__init__()
        self.nclasses = nclasses

    def call(self, inputs):
        x = tf.keras.layers.Conv2D(64, (2,2), activation='relu')(inputs)
        x = tf.keras.layers.AveragePooling2D()(x)
        x = tf.keras.layers.Conv2D(128, (2,2), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

        return output


