import cv2
import numpy
from matplotlib import pyplot as plt
#print(cv2.__version__) #check installation
img = cv2.imread('test.jpg') #image in same folder as script
px = img[300,400] #pixel coordinates (max in picture properties)
print(px) #prints BGR value (for grayscale only intensity)
print(img[300,400,0])
print(img[300,400,1])
print(img[300,400,2])
print(img.item(300,400,2))
img.itemset((300,400,2),0) #modify pixel values
print(img.item(300,400,2))
print(img.shape) #number of pixels and colour channels
print(img.size) #size of image
print(img.dtype)


imgL = cv2.imread('tsukuba_l.png',0) #left image
imgR = cv2.imread('tsukuba_r.png',0) #right image

stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
