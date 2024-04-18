import cv2
import numpy as np
 
 
gray_image = cv2.imread('images/blank.png',  cv2.IMREAD_GRAYSCALE)
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
np_img = np.asarray(gray_image)

height, width = np_img.shape

i, j = np.where(np_img == 0)

obstacleX = i.tolist()
obstacleY = j.tolist()




print(obstacleX)