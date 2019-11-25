import numpy as np
import scipy.io
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


'''
The lines below load the original .mat data files into Python.

info = scipy.io.loadmat(path+'info.mat')
picemo = info['picemo'] # (72, 1)
picphenos = info['picphenos'] # (72, 100)

with h5py.File(path+'indmaps100.mat', 'r') as allmaps:
    print(allmaps['allmaps'].shape)
'''

def input_function():
    
    path = os.path.dirname(os.path.abspath(__file__))

    mem_0 = scipy.io.loadmat(path+'/mem0.mat')['mem0_d10']
    mem_1 = scipy.io.loadmat(path+'/mem1.mat')['mem1_d10']
    
    labels_0 = np.zeros((len(mem_0), 1))
    labels_1 = np.ones((len(mem_1), 1))
    
    mem_labels_0 = np.concatenate((mem_0, labels_0), axis=1)
    mem_labels_1 = np.concatenate((mem_1, labels_1), axis=1)
    
    data = np.concatenate((mem_labels_0, mem_labels_1), axis=0)
    
    return data


data = input_function()

columns = [str(i) for i in range(0, data.shape[1])]

df = pd.DataFrame(data=data,
                  index=pd.RangeIndex(0,data.shape[0]),
                  columns=columns)


train, test = train_test_split(df, test_size=0.3)
train, val = train_test_split(train, test_size=0.3)


def df_to_ds(df, shuffle=True, batch_size=64):
    df = df.copy()
    labels = df.pop(str(df.shape[1]-1))
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


train_ds = df_to_ds(train)
val_ds = df_to_ds(val, shuffle=False)
test_ds = df_to_ds(test, shuffle=False)

feature_columns = []
for header in columns[0:64]:
    feature_columns.append(feature_column.numeric_column(header))
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(64, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=20)


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
