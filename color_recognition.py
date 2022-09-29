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

# img = cv2.imread('./newImages/09/12/autsheep_01_20220912092840.jpg')
# cropped_img = img[int(height * 0.38) : int(height - (height * 0.05)), int(width * 0.25) : int(width - (width * 0.10))]
# cv2.imshow('img', cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

create_data = False

if create_data:
    os.chdir('./newImages')

    images = []

    sc = StandardScaler()

    # Label Good Image and Bad Image. J stands for Month, whilst I represents Day.
    for j in range(8, 10):
        start = 0
        end = 0
        if j == 8:
            start = 13
            end = 30
        elif j == 9:
            start = 3
            end = 19

        os.chdir(f"./0{j}")
        for i in range(start, end):
            if (i > 17 and i < 22 and j == 8):
                continue
            os.chdir(f"./{i}")
            for f in os.listdir("."):
                print(f)
                if f.endswith(".jpg"):
                    src_img = cv2.imread(f)
                    
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

                    # If the month is equal to 8 and the day is more than the 17th, or the month is equal to September and the day is less than the 15th, then label the grass as 1 (good).
                    if j == 8 and i > 17 or j == 9 and i < 15:
                        label = 1

                    images.append({"Red": avg_col[0], "Green": avg_col[1], "Blue": avg_col[2], "Label": label})

            os.chdir('../')

        os.chdir('../')

    # TODO: Balance good grass image quantity and bad grass image quantitiy. 

    # Create Dataframe containing RGB values of image.
    df = pd.DataFrame(images)

    print(df.head())

    compression_opts = dict(method='zip', archive_name='out.csv')  

    # Convert dataframe to CSV/excel file.
    df.to_csv('out.zip', index=False, compression=compression_opts)  

# Getting weather statistics from a weather API (IPGeolocation.IO)
response = requests.get("https://api.ipgeolocation.io/astronomy?apiKey=73ccbbd73db442feb18aa714a4cdc43b&lat=36.8509&long=174.7645")

weather = dict(response.json())
 
print(weather.get("sunrise"), weather.get("sunset"))

# Creating the model
def createModel():
    model = Sequential()
    model.add(Dense(20, input_dim = 3, activation = 'relu', name = 'input'))
    model.add(Dense(60, activation = 'relu', name = 'hidden1'))
    model.add(Dense(60, activation = 'relu', name = 'hidden2'))
    model.add(Dense(2, activation = 'softmax', name = 'output'))
    print(model.summary())
    
    #Specify the loss function,optimiser and the evaluator.
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), metrics=['sparse_categorical_accuracy'])

    return model

# Get Data from csv
df = pd.read_csv('./dataset/out/out.csv')
df = df.sort_values("Label")

balanced = df[:372]
balanced = balanced.sample(frac=1)

# Train Test Split
from sklearn.model_selection import train_test_split

train, test = train_test_split(balanced, test_size=0.2)

# Merge all inputs into one array
merged_train = np.stack([train["Red"], train["Green"], train["Blue"]], axis=1)
merged_test = np.stack([test["Red"], test["Green"], test["Blue"]], axis=1)

# Train the model
model = createModel()

#Load previous model from 'path'
#model = tensorflow.keras.models.load_model('./model')

train = model.fit(merged_train, train["Label"], epochs = 500, batch_size = 64)

#Evaluate the model
y_pred = model.predict(merged_test)
  
print(y_pred)

pred = []

for i in y_pred:
    if i[1] < 0.5:
        pred.append(0)
    else:
        pred.append(1)    

# print(pred)

tp = np.array(pred)
fp = np.array(test["Label"])

combined = np.vstack((tp, fp)).T
combined

print(combined)

print(model.evaluate(merged_test, test["Label"]))

#Save Model to 'Path'
model.save('./model')
