import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import os

test_images={}
folder_dir = "../data/processed_images"
label_set=set()

def load_images_from_folder(folder, size=(32, 32)):
    images = []
    labels = []
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        label_set.add(class_name) 
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = Image.open(img_path)
                img = img.resize(size)
                img_array = np.array(img)
                images.append(img_array.reshape(32, 32, 1))
                labels.append(class_name)
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder(folder_dir)
le = LabelEncoder()
print(len(label_set))
le.fit(labels)
labels = le.transform(labels)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

train_images = train_images / 255.0
test_images = test_images / 255.0
val_images = val_images / 255.0

# data=[]
# labels=[]
# #print(labels)
# class_folders = []
# for folder in os.listdir("../data/processed_images"):
#     class_folders.append(folder)
    
# test=[]
# test_labels=[]
# train=[]
# train_labels=[]
# validation=[]
# val_labels=[]
# label_dict={}

# #assigning numbers to each class
# for i in range(len(class_folders)):
#     label_dict[class_folders[i]]=i

# for folder in class_folders:
#     samples=os.listdir("../data/processed_images/"+folder)
#     train+=samples[0:int(len(samples)*0.6)]
#     for i in range(int(len(samples)*0.6)):
#         train_labels.append(label_dict[folder])
#     test+=samples[int(len(samples)*0.6):int(len(samples)*0.8)]
#     for i in range(int(len(samples)*0.6), int(len(samples)*0.8)):
#         test_labels.append(label_dict[folder])
#     validation+=samples[int(len(samples)*0.8):]
#     for i in range(int(len(samples)*0.8), len(samples)):
#         val_labels.append(label_dict[folder])
# #print(test_labels)

# # train = tf.keras.utils.normalize(train, axis=1) #similar to dividing by 255 (but not equivalent in result)
# # test = tf.keras.utils.normalize(test, axis=1) #Also, don't know why we are using "axis=1" specifically, but that's what's normally used with image normalization
# # validate = tf.keras.utils.normalize(validation, axis=1) #Also, don't know why we are using "axis=1" specifically, but that's what's normally used with image normalization
# train_labels = np.array(train_labels)
# train = np.array(train_labels)

# test_labels = np.array(test_labels)
# val_labels=np.array(val_labels)


'''
The input layer is the first layer in the network and it is where the input data is fed into the network. 
The input layer does not perform any computation, it simply receives the input and passes it on to the next layer.
The hidden layers are the layers that come after the input layer and before the output layer. 
These layers perform the bulk of the computation in the network, such as feature extraction and abstraction. 
The output layer is the final layer in the network and it produces the output of the network.  
'''
 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(82))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

#model.save('model.keras')

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))