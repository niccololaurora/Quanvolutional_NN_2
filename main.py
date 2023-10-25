import tensorflow as tf
import numpy as np

from qclassifier import Quanvolutional_NN



def main():

    dimensione_filtro = 2
    numero_immagini = 1000
    depth = 10

    qcnn = Quanvolutional_NN(filterdim = dimensione_filtro, train_size = numero_immagini)
    qcnn.initialize_data(10)
    qcnn.quanvolutional_layer()
    history = qcnn.training_loop()
    qcnn.plot_metrics(history)

if __name__ == "__main__":
    main()

    