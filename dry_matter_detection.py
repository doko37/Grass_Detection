import tensorflow
import os
from datetime import datetime
import cv2
import numpy as np
import json

model = tensorflow.keras.models.load_model('./retrained-model')

height = 2560
width = 1440

time = datetime.now()

year = time.strftime("%Y")
month = time.strftime("%m")
day = time.strftime("%d")

# os.chdir("/home/kauricone/TeamSheep/SheepData/Images/{}/{}/{}/".format(year, month, day))
os.chdir("./newImages/10/2")

images = []

for f in os.listdir("."):
    print(f)
    if f.endswith(".jpg"):

        # Read in the image
        img = cv2.imread(f)

        # Crop the image
        cropped_img = img[int(height * 0.38) : int(height - (height * 0.05)), int(width * 0.25) : int(width - (width * 0.10))]

        # Calculate the average RGB values
        avg_col_row = np.average(cropped_img, axis=0)
        avg_col = np.average(avg_col_row, axis=0)

        # Normalize
        avg_col = avg_col/(255.0)

        # Append the raw data to array
        images.append([avg_col[0], avg_col[1], avg_col[2]])


y_pred = model.predict(images)
avg_pred = np.mean(y_pred)
max = np.max(y_pred)
min = np.min(y_pred)

results = [float(avg_pred), float(max), float(min)]

print(results)

# def write_json(data, filename="/home/kauricone/TeamSheep/ObjectDetection/results.json"):
#     with open(filename, 'r+') as file:
#         # read in json file to variable
#         file_data = json.loads(file.read())

#         # add new data to json file
#         file_data["results"][-1]["grass"] = data

#         # move write head
#         file.seek(0)

#         # write data to file
#         json.dump(file_data, file, indent=4)

# write_json(results)