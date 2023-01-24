


import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from PIL import Image as im
    

img = cv2.imread(r'C:\My_Data\M2M Data\save1\tensor(201).png',0) # Read in your image
temp = np.zeros(img.shape)
temp[np.where(img!=0)]=1
data = im.fromarray(temp).convert('L')

# Threshold and invert image as not actually binary
thresh = data.point(lambda p: p > 0)

# Get bounding box of thresholded image
bbox1 = thresh.getbbox()
crop1 = thresh.crop(bbox1)

# Invert and crop again
crop1n = ImageOps.invert(crop1)
bbox2  = crop1n.getbbox()
crop2  = crop1.crop(bbox2)  # You don't actually need this - it's just for debug

# Trim original, unthresholded, uninverted image to the two bounding boxes
result = data.crop(bbox1).crop(bbox2)

plt.figure()
plt.imshow(result)





