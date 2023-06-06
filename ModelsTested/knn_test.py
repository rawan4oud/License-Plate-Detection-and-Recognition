import pickle
import cv2
import joblib as jl
import time
import os

from Preprocessing.features import pixel_intensity


def returnLicenseNB():
    start_time = time.time()


    # Load the trained model and scaler
    model = pickle.load(open('../Saved Models/knn_model.sav', 'rb'))
    scaler = jl.load('../Extra/std_scaler.bin')
    with open('../Extra/encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)



    # Load and preprocess the test images
    result_list = ""
    for img_file in os.listdir("../Dataset/Segmented Images"):
        img_test = cv2.imread(os.path.join("../Dataset/Segmented Images", img_file))

        # Invert the colors
        inverted_img = cv2.bitwise_not(img_test)

        inverted_img = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(inverted_img, 0, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt0 = contours[0]

            # compute rectangle (minimum area)
            X, Y, W, H = cv2.boundingRect(cnt0)

            # crop image following the rectangle
            cropped_image = inverted_img[int(Y):int(Y + H), int(X):int(X + W)]

            # resize image
            cropped_image = cv2.resize(inverted_img, (16, 16), interpolation=cv2.INTER_AREA)


        else:
            cropped_image = cv2.resize(inverted_img, (16, 16), interpolation=cv2.INTER_AREA)


        # Reshape the image and scale it
        cropped_image = pixel_intensity(cropped_image).reshape(1, -1)
        cropped_image_scaled = scaler.transform(cropped_image)

        # Make the prediction
        pred = (model.predict(cropped_image_scaled))
        decoded_labels = label_encoder.inverse_transform((pred))
        result_list+=(str(decoded_labels[0]))

    print(result_list)
    end_time = time.time()
    #print(f"Testing time: {end_time-start_time:.2f} seconds")

    return result_list