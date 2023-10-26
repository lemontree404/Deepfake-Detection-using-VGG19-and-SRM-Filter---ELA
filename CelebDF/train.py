import os
import sys
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPool2D, Conv2D
from keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split
from keras.metrics import Recall,Precision,AUC
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import save_model,load_model
import matplotlib.pyplot as plt

data = []

real_dir = sys.argv[1]
fake_dir = sys.argv[2]
model_file = sys.argv[3]
model_name = sys.argv[4]

for i in [real_dir,fake_dir]:

    if i==real_dir:
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

test_db.to_csv(f"{model_name}_test_db")

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
              metrics=['accuracy',Precision(),Recall(),AUC()],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                              mode = 'max' , 
                              patience = 5,
                              verbose = 1)

callback_list = [earlystopping]

hist = model.fit(train_imgs,
                    validation_data=val_imgs,
                    epochs = 5,
                    callbacks = callback_list)

if "models" not in os.listdir():
    os.mkdir("models")

save_model(model,f"models/{model_file}")

if model_name not in os.listdir():
    os.mkdir(model_name)

fig,ax = plt.subplots(2,1)
ax[0].plot(hist.history["auc"])
ax[0].set_title("Val. AUC")

ax[1].plot(hist.history["accuracy"])
ax[1].set_title("Val. Accuracy")

fig.suptitle(f"{model_name} Metrics")
fig.set_tight_layout(True)
fig.savefig(f"{model_name}/metrics",dpi=1000)