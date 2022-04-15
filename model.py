# -*- coding: utf-8 -*-
"""Tensorflow Learning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XXkS1Wf_BrNgOO04IIXf5jaNfq0rAd5g
"""

import tensorflow as tf 
import numpy as np 
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import cv2
import PIL.Image as Image



!git clone https://github.com/chandrikadeb7/Face-Mask-Detection

import glob

imgs = []
for path in glob.glob("/content/Face-Mask-Detection/dataset/with_mask/*"):
    img = Image.open(path)
    img = img.resize((200,200))
    imgs.append(np.array(img))

im = []
for img in imgs:
    if img.shape[-1]!=3:
        pass
    else:
        im.append(img)

mask= np.array(im)



imgs = []
for path in glob.glob("/content/Face-Mask-Detection/dataset/without_mask/*"):
    img = Image.open(path)
    img = img.resize((200,200))
    imgs.append(np.array(img))

no_mask= np.array(imgs)

print(mask.shape)
print(no_mask.shape)

ones= np.ones(2052)
zeros= np.zeros(1930)

features = np.append(mask, no_mask, axis=0)
features = features/255.0
labels = np.append(ones, zeros, axis=0)

labels

plt.imshow(features[1])

plt.imshow(features[-1])

features.shape

features[2]



model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(64,kernel_size=(3,3), input_shape=(200,200,3), activation="relu"),
                             tf.keras.layers.MaxPool2D((2,2)),
                             tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

lbl = tf.keras.utils.to_categorical(labels)

with tf.device("/GPU:0"):
    model.fit(features, lbl, epochs=10)

model.save("MyModel")