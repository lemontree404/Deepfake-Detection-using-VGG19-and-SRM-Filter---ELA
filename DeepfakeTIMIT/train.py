import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPool2D, Conv2D
from keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split
from keras.metrics import Recall,Precision
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

data = []

for i in ["real_images_srm","fake_images_srm"]:

    if i=="real_images_srm":
        label = 'real'
    else:
        label = 'fake'

    data += [[f"{i}/{j}",label] for j in os.listdir(i)]


db = pd.DataFrame(data,columns=["path","label"])
db = db.sample(frac=1)
db.reset_index(inplace=True)
db.drop("index",axis=1,inplace=True)


train_db, test_db = train_test_split(db, test_size=0.2, random_state=2, shuffle=True)

train_generator = ImageDataGenerator(validation_split=0.2)
test_generator = ImageDataGenerator()

train_imgs = train_generator.flow_from_dataframe(
    dataframe = train_db,
    x_col = "path",
    y_col = "label",
    target_size = (128, 128),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 32,
    shuffle = True,
    subset = "training"
)

val_imgs = train_generator.flow_from_dataframe(
    dataframe = train_db,
    x_col = "path",
    y_col = "label",
    target_size = (128, 128),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 32,
    shuffle = True,
    subset = "validation"
)

test_imgs = test_generator.flow_from_dataframe(
    dataframe = test_db,
    x_col = "path",
    y_col = "label",
    target_size = (128, 128),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 32,
    shuffle = False
)


base_model = VGG19(input_shape=(128, 128, 3), 
                         include_top=False,
                         weights="imagenet")

for layer in base_model.layers:
    layer.trainable=False

model=Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512,"relu",kernel_initializer="he_uniform"))
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              metrics=['accuracy',Precision(),Recall()],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                              mode = 'max' , 
                              patience = 5,
                              verbose = 1)

callback_list = [earlystopping]

hist = model.fit(train_imgs,
                    validation_data=val_imgs,
                    epochs = 10,
                    callbacks = callback_list)


plt.plot(hist.history["val_loss"])
plt.plot(hist.history["loss"])
y_pred = model.predict(test_imgs).argmax(axis=1)
print(classification_report(test_imgs.labels,y_pred))