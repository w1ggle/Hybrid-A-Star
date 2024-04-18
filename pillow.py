from PIL import Image
import numpy as np
 
 
img= Image.open("blank.png").convert('L')
np_img = np.asarray(img)
 
height, width = np_img.shape

i, j = np.where(np_img == 0)

obstacleX = i.tolist()
obstacleY = j.tolist()




print(obstacleX)