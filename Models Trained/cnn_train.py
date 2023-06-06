import os
import glob
import cv2
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time

paths ={
    'DATASET_PATH': os.path.join('../Dataset', 'Dataset'),
    'NEW_DATASET_PATH': os.path.join('../Dataset', 'New CNN Dataset')
}

categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
              'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
data = []

# Read the input images
for category in categories:
    path = os.path.join(paths['DATASET_PATH'], category)
    label = categories.index(category)

    for img in glob.glob(path + "\\*.png"):
        if (img is not None):
            img0 = cv2.imread(img)

            # Convert the image to grayscale
            img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cnt0 = contours[0]

                # Compute rectangle (minimum area)
                X, Y, W, H = cv2.boundingRect(cnt0)

                # Crop image following the rectangle
                cropped_image = img1[int(Y):int(Y + H), int(X):int(X + W)]

                # Resize image
                cropped_image = cv2.resize(cropped_image, (16, 16), interpolation=cv2.INTER_AREA)

                # Substring for image name
                newImage = img[14:]

                new_path = os.path.join(paths['NEW_DATASET_PATH'], str(category))
                cv2.imwrite(os.path.join(new_path, newImage), cropped_image)

                # Append the cropped image and the label to the data list
                data.append([cropped_image, label])

            else:
                newImage = img[14:]
                new_path = os.path.join(paths['NEW_DATASET_PATH'], str(category))
                cropped_image = cv2.resize(img1, (16, 16), interpolation=cv2.INTER_AREA)
                data.append([cropped_image, label])
                cv2.imwrite(os.path.join(new_path, newImage), cropped_image)

print(len(data))

#write data into pickle file
pick_in = open('../Preprocessing/data.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

#get data from pickle file
pick_in = open('../Preprocessing/data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

# Shuffle the data
random.shuffle(data)
print(data[0])

# Separate the features and labels
X = []
y = []
for feature, label in data:
    X.append(feature)
    y.append(label)

# Convert the features and labels to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape the features
X = X.reshape(-1, 16, 16, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=36)
y_test = to_categorical(y_test, num_classes=36)

model = Sequential()

model.add(Conv2D(36, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

# Train the model
model.fit(X_train, y_train, batch_size=8, epochs=100, verbose=1, validation_data=(X_test, y_test))

end_time = time.time()

print(f"Training time: {end_time-start_time:.2f} seconds")

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('cnn.h5')

