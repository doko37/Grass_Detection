from audioop import avg
from codecs import ignore_errors
from telnetlib import SE
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import sys
import colorsys
import requests
import pandas as pd
import tensorflow
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from IPython.display import display

width, height = Image.open('./newImages/08/13/autsheep_01_20220813134306.jpg').size

images = []

src_img = cv2.imread('./newImages/autsheep_01_20220924101638.jpg')

# Crop images to cut out any unnecessary data
cropped_img = src_img[int(height * 0.38) : int(height - (height * 0.05)), int(width * 0.25) : int(width - (width * 0.10))]

# Calculate the average RGB value of the entire cropped image.
avg_col_row = np.average(cropped_img, axis=0)
avg_col = np.average(avg_col_row, axis=0)

print(avg_col)

# Normalize Average RGB Values.
avg_col = avg_col/(255.0)
print(avg_col)
label = 0

images.append({"Red": avg_col[0], "Green": avg_col[1], "Blue": avg_col[2], "Label": label})

model = tensorflow.keras.models.load_model('./model')

test = pd.DataFrame(images)

merged_test = np.stack([test["Red"], test["Green"], test["Blue"]], axis=1)

print(model.predict(merged_test))
