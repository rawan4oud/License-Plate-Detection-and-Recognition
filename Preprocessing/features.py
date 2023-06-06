import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize

###########################################################################################################################################
#features
###########################################################################################################################################

def pixel_intensity(cropped_image):
    n_samples = len(cropped_image)
    data = cropped_image.reshape((n_samples, -1))
    return data

def sobel_edge(cropped_image):
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    return sobelxy

def canny_edge(cropped_image):
    edges = 255-cv2.Canny(image=cropped_image, threshold1=100, threshold2=200) # Canny Edge Detection
    return edges

def histogram(cropped_image):
    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate horizontal projection
    hor_proj = np.sum(cropped_image, axis=1)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width), np.uint8)

    # Draw a line for each row
    for idx, value in enumerate(hor_proj):
        cv2.line(blankImage, (0, idx), (width-int(value), idx), (255, 255, 255), 1)

    # Save result
    blankImage = cv2.resize(blankImage, (8, 8), interpolation=cv2.INTER_AREA)

    return blankImage

def histogram2(cropped_image):
    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate vertical projection
    hor_proj = np.sum(cropped_image, axis=0)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width), np.uint8)

    # Draw a line for each column
    for idx, value in enumerate(hor_proj):
        cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255, 255, 255), 1)

    # Save result
    blankImage = cv2.resize(blankImage, (128, 128), interpolation=cv2.INTER_AREA)

    return blankImage

def HOG(cropped_image):

    img1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)


    # resizing image
    resized_img = resize(thresh, (8 * 4, 8 * 4))

    # creating hog features
    fd, hog_image = hog(cropped_image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)

    hog_image = cv2.resize(cropped_image, (16, 16), interpolation=cv2.INTER_AREA)
    return hog_image

def sift(cropped_image):

    # Create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect keypoints and extract descriptors
    keypoints, descriptors = sift.detectAndCompute(sift, None)

    # Draw keypoints on the input image
    img_with_keypoints = cv2.drawKeypoints(cropped_image, keypoints, None)

    return img_with_keypoints

def LBP(cropped_image):

    # Apply LBP
    radius = 1
    n_points = 8 * radius
    lbp = np.zeros_like(cropped_image)
    for i in range(radius, cropped_image.shape[0] - radius):
        for j in range(radius, cropped_image.shape[1] - radius):
            center_pixel = cropped_image[i, j]
            pixel_values = [cropped_image[i - radius, j - radius], cropped_image[i - radius, j], cropped_image[i - radius, j + radius],
                            cropped_image[i, j + radius], cropped_image[i + radius, j + radius], cropped_image[i + radius, j],
                            cropped_image[i + radius, j - radius], cropped_image[i, j - radius]]
            binary_values = np.array(pixel_values) >= center_pixel
            binary_values = np.uint8(binary_values)
            binary_string = ''.join([str(b) for b in binary_values])
            decimal_value = int(binary_string, 2)
            lbp[i, j] = decimal_value

    return lbp



###########################################################################################################################################