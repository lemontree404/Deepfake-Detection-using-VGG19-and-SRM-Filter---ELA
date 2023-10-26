import os
import sys
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.metrics import AUC

srm_test_db = pd.read_csv("srm_model_test_db")
ela_test_db = pd.read_csv("ela_model_test_db")

test_generator = ImageDataGenerator()

test_imgs_srm = test_generator.flow_from_dataframe(
    dataframe = srm_test_db,
    x_col = "path",
    y_col = "label",
    target_size = (128, 128),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 32,
    shuffle = False
)

test_imgs_ela = test_generator.flow_from_dataframe(
    dataframe = ela_test_db,
    x_col = "path",
    y_col = "label",
    target_size = (128, 128),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 32,
    shuffle = False
)

srm_model = load_model("models/srm_model.h5")
ela_model = load_model("models/ela_model.h5")

y_pred_srm = srm_model.predict(test_imgs_srm).argmax(axis=1)
y_pred_ela = ela_model.predict(test_imgs_ela).argmax(axis=1)

print("------------------SRM Classifier----------------")
print(classification_report(test_imgs_srm.labels,y_pred_srm))
print("\n\n")

print("------------------ELA Classifier----------------")
print(classification_report(test_imgs_ela.labels,y_pred_ela))
print("\n\n")