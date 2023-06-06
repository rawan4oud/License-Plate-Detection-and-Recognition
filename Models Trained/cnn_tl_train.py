import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model



#cnn transfer learning

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
            img0 = cv2.resize(img0, (32, 32), interpolation=cv2.INTER_AREA)

             # Append the resized image and the label to the data list
            data.append([img0, label])

# Separate the input and labels
X = []
y = []
for input, label in data:
    X.append(input)
    y.append(label)

# Convert the features and labels to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape the features
X = X.reshape(-1, 32, 32, 3)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode the labels
y_train = to_categorical(y_train, num_classes=36)
y_test = to_categorical(y_test, num_classes=36)

# Load pre-trained ResNet50 model without top layers
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze all layers in the pre-trained model
for layer in resnet50.layers:
    layer.trainable = False

# Add new top layers to the pre-trained model
x = Flatten()(resnet50.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(36, activation='softmax')(x)

# Define the new model
model = Model(inputs=resnet50.input, outputs=output)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, y_test))

#Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
