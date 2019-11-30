import numpy as np
import scipy.io
import os
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # turn off irrelevant warning      

'''
The lines below load the original .mat data files into Python.

info = scipy.io.loadmat(path+'info.mat')
picemo = info['picemo'] # (72, 1)
picphenos = info['picphenos'] # (72, 100)

with h5py.File(path+'indmaps100.mat', 'r') as allmaps:
    print(allmaps['allmaps'].shape)
'''


def input_function():
    "Prepare dataset in shape of (observations, features)"
    path = os.path.dirname(os.path.abspath(__file__))
    mem_0 = scipy.io.loadmat(path+'/mem0.mat')['mem0_d10']
    mem_1 = scipy.io.loadmat(path+'/mem1.mat')['mem1_d10']
    labels_0 = np.zeros((len(mem_0), 1))
    labels_1 = np.ones((len(mem_1), 1))
    mem_labels_0 = np.concatenate((mem_0, labels_0), axis=1)
    mem_labels_1 = np.concatenate((mem_1, labels_1), axis=1)
    data = np.concatenate((mem_labels_0, mem_labels_1), axis=0)
    columns = [str(i) for i in range(0, data.shape[1])]
    df = pd.DataFrame(data=data,
                      index=pd.RangeIndex(0,data.shape[0]),
                      columns=columns,
                      dtype=np.float32)
    return df, columns

df, columns = input_function()


def df_to_ds(df, shuffle=True, batch_size=64):
    "Transform Pandas DataFrame to Tensorflow Dataset"
    df = df.copy()
    labels = df.pop(str(df.shape[1]-1))
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


def model(train, val, test, columns):
    "Building, compiling, fitting, and evaluating model"
    feature_columns = []
    for header in columns[0:len(columns)-1]:
        feature_columns.append(feature_column.numeric_column(header))
    feature_layer = layers.DenseFeatures(feature_columns)

    model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train,
              validation_data=val,
              epochs=7)

    loss, accuracy = model.evaluate(test)
    print("Accuracy", accuracy)


train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

train_ds = df_to_ds(train)
val_ds = df_to_ds(val)
test_ds = df_to_ds(test)

acc = model(train_ds, val_ds, test_ds, columns)