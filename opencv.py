import cv2
import numpy as np
 
image = cv2.imread('images/test.png', cv2.IMREAD_GRAYSCALE)
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
down_width = 300
down_height = 200
down_points = (down_width, down_height)
image = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
#cv2.imshow('Resized Up image by defining scaling factor', image)
#cv2.waitKey()

np_img = np.asarray(image)

height, width = np_img.shape

print(height)
print(width)