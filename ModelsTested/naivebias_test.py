import pickle
import cv2
import joblib as jl
import time

start_time = time.time()

model = pickle.load(open('../Saved Models/nb_model.sav', 'rb'))
scaler = jl.load('../Extra/std_scaler.bin')

# Load and preprocess the test image
img_test = cv2.imread("CNN Dataset/Segmented Images/")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_test, 0, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    cnt0 = contours[0]

    # compute rectangle (minimum area)
    X, Y, W, H = cv2.boundingRect(cnt0)

    # crop image following the rectangle
    cropped_image = img_test[int(Y):int(Y + H), int(X):int(X + W)]

    # resize image
    cropped_image = cv2.resize(img_test, (16, 16), interpolation=cv2.INTER_AREA)

else:
    cropped_image = cv2.resize(img_test, (16, 16), interpolation=cv2.INTER_AREA)


X_test = scaler.transform(cropped_image.flatten().tolist())


# Reshape the image and scale it
prediction = model.predict(X_test.reshape(-1, 1))

# Make the prediction
pred = (model.predict(prediction))

print(pred)

end_time = time.time()
print(f"Testing time: {end_time-start_time:.2f} seconds")

