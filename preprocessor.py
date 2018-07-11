import cv2
from sys import argv

img = cv2.imread(argv[1])
cv2.imshow('note', img)
cv2.waitKey(0)

b = -30. # brightness
c = 190.  # contrast
img = cv2.addWeighted(img, 1. + c/127., img, 0, b-c)
cv2.imshow('note',img)
cv2.waitKey(0)
cv2.imwrite(argv[1]+'boosted.jpg',img)
