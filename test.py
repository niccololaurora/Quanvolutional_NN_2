import tensorflow as tf
import numpy as np

from qclassifier import Quanvolutional_NN

def main():
    
    qcnn = Quanvolutional_NN()
    qcnn.initialize_data(10)
    qcnn.barplot()

if __name__ == "__main__":
    main()