

import pandas as pd 
import numpy as np
data=pd.read_csv('C:\\Users\\Admin\\Downloads\\sample_submission.csv')
data.head()
data.shape
data.isnull().sum()
data.duplicated().sum()
data.info()
import tensorflow as tf
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test= x_train/255.0, x_test/255.0
import matplotlib.pyplot as plt

def plot_sample(images, labels, num_rows=2, num_cols=5):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Assuming x_train and y_train are defined
plot_sample(x_train, y_train)

"""import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Display a sample image
plt.imshow(x_train[9], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()
"""

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train[...,np.newaxis], y_train,epochs=5,validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test[..., np.newaxis], y_test)
print('Test accuracy:', test_acc)


import gradio as gr

import numpy as np 
import gradio as gr
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image

from PIL import Image
import numpy as np

def preprocess_image(image):
    """Preprocess the image to be compatible with the MNIST model input.
    
    Parameters:
    image (PIL.Image or np.ndarray): The input image to be preprocessed.
    
    Returns:
    np.ndarray: The preprocessed image.
    """
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Convert to greyscale    
    image = image.convert('L')
    
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0 
    
    # Reshape to match the input shape of the model
    image = image.reshape(1, 28, 28)
    
    return image

# Prediction function
def predict_digit(image):
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make the prediction
    prediction = model.predict(image)
    
    return int(np.argmax(prediction))  # Ensure the output is a plain Python integer

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(image_mode='RGB'),
    outputs='label',
    live=True,
    title='MNIST Digit Classifier',
    description='Draw a digit (0-9) and see the prediction'
)

interface.launch()