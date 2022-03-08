
from tkinter import filedialog
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import matplotlib.pyplot as plt5
import matplotlib.image as mpimg
import datetime, os
from PIL import Image, ImageTk
import seaborn as sns
import cv2
import random
import os
import imageio
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE

import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.applications import resnet
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.callbacks import TensorBoard
import datetime
import time
from sklearn import metrics


directory = r'E:\Grad2\The IQ-OTHNCCD lung cancer dataset for detection'
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']


size_data = {}
for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    temp_dict = {}
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        height, width, channels = imageio.imread(filepath).shape
        if str(height) + ' x ' + str(width) in temp_dict:
            temp_dict[str(height) + ' x ' + str(width)] += 1 
        else:
            temp_dict[str(height) + ' x ' + str(width)] = 1
    
    size_data[i] = temp_dict
        



#Image Preprocessing and Testing

img_size = 256
for i in categories:
    cnt, samples = 0, 3
    fig, ax = plt.subplots(samples, 3, figsize=(15, 15))
    fig.suptitle(i)
    
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for curr_cnt, file in enumerate(os.listdir(path)):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, 0)
        
        img0 = cv2.resize(img, (img_size, img_size))
        
        img1 = cv2.GaussianBlur(img0, (5, 5), 0)
        
        ax[cnt, 0].imshow(img)
        ax[cnt, 1].imshow(img0)
        ax[cnt, 2].imshow(img1)
        cnt += 1
        if cnt == samples:
            break
        
plt.show()


data = []
img_size = 256

for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, 0)
        img = cv2.resize(img, (img_size, img_size))
        data.append([img, class_num])   
        
random.shuffle(data)
X, y = [], []
for feature, label in data:
    X.append(feature)
    y.append(label)
print("fffff")
print(y)
print('X length:', len(X))
print('y counts:', Counter(y))


X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0
print(X)
y = np.array(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size = 0.2, random_state=10, stratify=y)

print(len(X_train), X_train.shape)
print(len(X_valid), X_valid.shape)


#Applying SMOTE to oversample the data
print(Counter(y_train), Counter(y_valid))


print(len(X_train), X_train.shape)

X_train = X_train.reshape(X_train.shape[0], img_size*img_size*1)

print(len(X_train), X_train.shape)


print('Before SMOTE:', Counter(y_train))
smote = SMOTE()
X_train_sampled, y_train_sampled = smote.fit_resample(X_train, y_train)



X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
X_train_sampled = X_train_sampled.reshape(X_train_sampled.shape[0], img_size, img_size, 1)


print(len(X_train), X_train.shape)
print(len(X_train_sampled), X_train_sampled.shape)


NAME = "Lung-cancer-prediction-and-detection"

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


model1 = Sequential()

model1.add(Conv2D(64, (3, 3), input_shape=X_train_sampled.shape[1:]))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))


model1.add(Flatten())
model1.add(Dense(16))
model1.add(Dense(3, activation='softmax'))         

model1.summary()



loss_fn = keras.losses.SparseCategoricalCrossentropy()
model1.compile(loss=loss_fn, optimizer='Adagrad', metrics=['accuracy'])
#categorical_crossentropy
#MeanSquaredError
#squared_hinge
#binary_crossentropy

history = model1.fit(X_train_sampled, y_train_sampled, batch_size=0, epochs=200, validation_data=(X_valid, y_valid))


y_pred = model1.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print('dddddddssssssssssssssssssssss')
print(y_pred_bool)
print(len(y_valid))


print(model1.evaluate(X_train_sampled,y_train_sampled, batch_size=8))

print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
print(confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))




plt2.plot(history.history['accuracy'], label='Train')
plt2.plot(history.history['val_accuracy'], label='Validation')
plt2.title('Model Accuracy')
plt2.ylabel('Accuracy')
plt2.xlabel('Epoch')
plt2.legend()
plt2.show()

plt3.plot(history.history['loss'], label='Train')
plt3.plot(history.history['val_loss'], label='Validation')
plt3.title('Model Loss')
plt3.ylabel('Loss')
plt3.xlabel('Epoch')
plt3.legend()
plt3.show()



print('If you want to make sure about you result you can upload you CT Scan \n'+'Press Y if you want and N if no')
Answer = input()
if(Answer == 'y'):
    Answer=1
else:
    Answer=0

if(Answer == 1):
    ImgPath = filedialog.askopenfilename(initialdir = "/",title = "Select an image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
n=10
while n > 0:
 img_size = 256
 UserImg = cv2.imread(ImgPath, 0)
 UserImg = cv2.resize(UserImg, (img_size, img_size))
 plt4.title('Your CT Scan')
 plt4.imshow(UserImg)
 plt4.show()
 
 YourImg = []
 YourImg.append(UserImg)
 YourImg = np.array(YourImg).reshape(-1, img_size, img_size, 1)
 YourImg = YourImg / 255.0
 
 
 
 y_pred = model1.predict(YourImg, verbose=1)
 y_pred_bool = np.argmax(y_pred, axis=-1)
 score = model1.evaluate(X_valid,y_valid, batch_size=8)
 print('The accuracy is ',score)
 
 score[1]=str(score[1]*100)
 if(y_pred_bool==0):
     print('This CT scan is a Benign case with accuracy '+score[1]+'%')
 if(y_pred_bool==1):
     print('This CT scan is a Malignant case with accuracy '+score[1]+'%')
 if(y_pred_bool==2):
     print('This CT scan is a Normal case with accuracy '+score[1]+'%')
 
 print('If you want to check againg press Y if not press N')
 Answer = input()
 if(Answer == 'y'):
     Answer=1
 else:
     Answer=0
 
 if(Answer == 1):
     ImgPath = filedialog.askopenfilename(initialdir = "/",title = "Select an image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
 
 if(Answer == 0):
     break
 






