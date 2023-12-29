import cv2
import numpy as np

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_image_contours(image):
    contours = find_contours(image)
    height, width = image.shape[:2]
    blank_image = np.zeros((height,width,3), np.uint8)
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        traffic_sign = image[y:y+h, x:x+w]
        blank_image = traffic_sign
    return blank_image

def getThreshold(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_white = np.array([0,0,200], dtype=np.uint8)
    # upper_white = np.array([180,30,255], dtype=np.uint8)
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_red1 = np.array([0,100,100], dtype=np.uint8)
    upper_red1 = np.array([10,255,255], dtype=np.uint8)

    lower_red2 = np.array([170,100,100], dtype=np.uint8)
    upper_red2 = np.array([180,255,255], dtype=np.uint8)

    for i in range(len(image)):
        for j in range(len(image[i])):
            if (image[i][j] == [255,255,255]).all():
                continue
            elif ((image[i][j] >= lower_red1).all() and (image[i][j] <= upper_red1).all()) or (image[i][j] >= lower_red2).all() and (image[i][j] <= upper_red2).all():
                image[i][j] = [255,0,0]
            else:
                image[i][j] = [0,0,0]
    return image
image = cv2.imread('./stop.png')
contours = find_contours(image)
blank_image = get_image_contours(image)
new_image = getThreshold(blank_image)
cv2.imshow('Detected Signs', new_image)
cv2.waitKey(0)