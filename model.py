import pandas as pd
import numpy as np
import io
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import cv2
from keras.layers.normalization import BatchNormalization
import datetime, os
from keras.models import load_model

num_features = 64
num_labels = 7
batch_size = 64
epochs = 40
width, height = 48, 48

emotion_data = pd.read_csv('/content/drive/MyDrive/Emotion_Detection/fer2013.csv')
print(emotion_data)
print(emotion_data.dtypes)
X_train = []
y_train = []
X_test = []
y_test = []
pixels = emotion_data['pixels'].tolist() # 1

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # 2
    face = np.asarray(face).reshape(width, height) # 3
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1) # 6

emotions = pd.get_dummies(emotion_data['emotion']).values # 7

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

#model = Sequential()
#model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
#model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2,2), strides=(2,2)))

#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(128, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(128, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(256, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(256, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(256, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

#model.add(Flatten())
#model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(7, activation='softmax'))

#model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(X_train,y_train,batch_size=32,epochs=30,verbose=1,validation_data=(X_test, y_test))
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))
model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = tf.keras.callbacks.ModelCheckpoint('/content/Untitled Folder', monitor='val_loss', verbose=1, save_best_only=True)
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_test), np.array(y_test)),
          shuffle=True,
          callbacks=[lr_reducer, tensorboard_callback, early_stopper, checkpointer])
