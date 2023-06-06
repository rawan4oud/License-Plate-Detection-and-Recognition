import os
import cv2
import glob
import pickle

from Preprocessing.features import pixel_intensity

paths ={
    'DATASET_PATH': os.path.join('../Dataset', 'Knn-SVM-NB Dataset'),
}

categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
              'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print(len(categories))
data = []


#read the input images
for category in categories:
    path = os.path.join(paths['DATASET_PATH'], category)

    label = category

    for img in glob.glob(path + "\\*.png"):
        if (img is not None):

            img0 = cv2.imread(img)

            # convert the image to grayscale
            img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cnt0 = contours[0]

                # compute rectangle (minimum area)
                X, Y, W, H = cv2.boundingRect(cnt0)

                # crop image following the rectangle
                cropped_image = img1[int(Y):int(Y + H), int(X):int(X + W)]

                # resize image
                cropped_image = cv2.resize(cropped_image, (16, 16), interpolation=cv2.INTER_AREA)

                # substring for image name
                newImage = img[14:]

                #new_path = os.path.join(paths['NEW_DATASET_PATH'], str(category))
                #cv2.imwrite(os.path.join(new_path, newImage), cropped_image)

                data.append([pixel_intensity(cropped_image), label])

            else:
                newImage = img[14:]
                #new_path = os.path.join(paths['NEW_DATASET_PATH'], str(category))
                cropped_image = cv2.resize(img1, (16, 16), interpolation=cv2.INTER_AREA)
                data.append([pixel_intensity(cropped_image), label])

                #cv2.imwrite(os.path.join(new_path, newImage), cropped_image)

print(len(data))


#write data into pickle file
pick_in = open('data.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()