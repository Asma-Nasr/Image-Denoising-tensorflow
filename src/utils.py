
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

def load():
    # Load and preprocess MNIST data
    (x_train, _), (x_test, _) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Reshape to (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test

def make_noise(x_train, x_test, noise_factor=0.5):
    # Add noise to the images
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    # Clip values to [0, 1]
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train_noisy, x_test_noisy


def show_images(x_test_noisy, denoised_images, x_test, n=10):
    """Visualize original noisy, denoised, and original clean images."""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original noisy images
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        plt.title('Noisy')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display denoised images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(denoised_images[i].reshape(28, 28))
        plt.gray()
        plt.title('Denoised')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display original clean images
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        plt.title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_history(history):
    """Plot training and validation loss history."""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
