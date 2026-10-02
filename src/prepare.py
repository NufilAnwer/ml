import os
import numpy as np
import tensorflow as tf

def prepare():
    os.makedirs('data/raw', exist_ok=True)
    
    # Load dataset directly from keras
    print("Downloading Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Save raw arrays
    np.savez_compressed('data/raw/train.npz', x=x_train, y=y_train)
    np.savez_compressed('data/raw/test.npz', x=x_test, y=y_test)
    print("Raw data saved to data/raw/")

if __name__ == '__main__':
    prepare()
