from keras.models import load_model
import cv2
import numpy as np
import time
import os


def returnLicenseNB():
    start_time = time.time()

    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                  'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    result_list = ""
    for img_file in os.listdir("Dataset/Segmented Images"):
        img_test = cv2.imread(os.path.join("Dataset/Segmented Images", img_file))
        # convert the image to grayscale
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

        cropped_image = (cropped_image.reshape(-1, 16, 16, 1))/255.0


        model = load_model('Saved Models/cnn.h5')
        pred = model.predict(cropped_image, verbose=0)
        label = str(categories[np.argmax(pred)])
        result_list+=label

    #print(result_list)
    end_time = time.time()
    #print(f"Testing time: {end_time-start_time:.2f} seconds")

    return result_list
