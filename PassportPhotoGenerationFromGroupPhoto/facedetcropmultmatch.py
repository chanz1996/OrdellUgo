#facedetect,crop(save detected faces to a file) and then multiple template matching with specified cropped image
import cv2
import numpy as np
from matplotlib import pyplot as plt

#import sys
count=0
# Get user supplied values
imagePath = 'abba.png'
cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #extract detected faces
    crop_img=image[y:y+h, x:x+w]
    cv2.imwrite('crop_img%d.jpg'%count,crop_img)
    count+=1

#image = image[c1:c1+25,r1:r1+25]
#multiple tempate matching
img_rgb = cv2.imread('abba.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('crop_img3.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
 cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv2.imwrite('res.png',img_rgb)
cv2.imshow("Faces found", image)
cv2.waitKey(0)

