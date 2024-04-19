import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


import os

test_images={}
folder_dir = "../data/processed_images"

def load_from_folder(folder):
    images=[]
    for symbol in os.listdir(folder_dir):
        imgs = Image.open(os.path.join(folder_dir, symbol))
        if imgs is not None:
            images.append(np.array(imgs))
    return images

data=[]
labels=[]
class_folders = []
for folder in os.listdir("../data/processed_images"):
    class_folders.append(folder)
    
test=[]
test_labels=[]
train=[]
train_labels=[]
validation=[]
val_labels=[]
for folder in class_folders:
    samples=os.listdir("../data/processed_images/"+folder)
    train+=samples[0:int(len(samples)*0.6)]
    for i in range(int(len(samples)*0.6)):
        train_labels+=folder
    test+=samples[int(len(samples)*0.6):int(len(samples)*0.8)]
    for i in range(int(len(samples)*0.6), int(len(samples)*0.8)):
        test_labels+=folder
    validation+=samples[int(len(samples)*0.8):]
    for i in range(int(len(samples)*0.8), len(samples)):
        val_labels+=folder


'''
The input layer is the first layer in the network and it is where the input data is fed into the network. 
The input layer does not perform any computation, it simply receives the input and passes it on to the next layer.
The hidden layers are the layers that come after the input layer and before the output layer. 
These layers perform the bulk of the computation in the network, such as feature extraction and abstraction. 
The output layer is the final layer in the network and it produces the output of the network.  
'''

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.train_()

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))