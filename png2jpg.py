#!/usr/bin/env python
from glob import glob                                                           
import cv2 
pngs = glob('./data/**/*.png')

for j in pngs:
    img = cv2.imread(j)
    print('converting : '+j)
    cv2.imwrite(j[:-3] + 'jpg', img)