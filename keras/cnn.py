import tensorflow as tf
import pickle
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
#print(labels)
class_folders = []
for folder in os.listdir("../data/processed_images"):
    class_folders.append(folder)
    
test=[]
test_labels=[]
train=[]
train_labels=[]
validation=[]
val_labels=[]
label_dict={}

#assigning numbers to each class
for i in range(len(class_folders)):
    label_dict[class_folders[i]]=i

for folder in class_folders:
    samples=os.listdir("../data/processed_images/"+folder)
    train+=samples[0:int(len(samples)*0.6)]
    for i in range(int(len(samples)*0.6)):
        train_labels.append(label_dict[folder])
    test+=samples[int(len(samples)*0.6):int(len(samples)*0.8)]
    for i in range(int(len(samples)*0.6), int(len(samples)*0.8)):
        test_labels.append(label_dict[folder])
    validation+=samples[int(len(samples)*0.8):]
    for i in range(int(len(samples)*0.8), len(samples)):
        val_labels.append(label_dict[folder])
#print(test_labels)

# train = tf.keras.utils.normalize(train, axis=1) #similar to dividing by 255 (but not equivalent in result)
# test = tf.keras.utils.normalize(test, axis=1) #Also, don't know why we are using "axis=1" specifically, but that's what's normally used with image normalization
# validate = tf.keras.utils.normalize(validation, axis=1) #Also, don't know why we are using "axis=1" specifically, but that's what's normally used with image normalization
train_labels = np.array(train_labels)
train = np.array(train_labels)

test_labels = np.array(test_labels)
val_labels=np.array(val_labels)


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

model.fit(train, train_labels, epochs=5)

model.train_()

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))