import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('Desk_L.JPG',0)
img2 = cv.imread('Desk_R.JPG',0)

# Initiate ORB detector
orb = cv.ORB_create(edgeThreshold=5, patchSize=30, nlevels=5, fastThreshold=10, scaleFactor=1.2, WTA_K=2,
                    scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=15000)
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

# draw only keypoints location,not size and orientation
res1 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=6)
res2 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=6)

plt.imshow(res1), plt.show()
#plt.imsave('ORB1', res1)
plt.imshow(res2), plt.show()
#plt.imsave('ORB2', res2)


#This is our BFMatcher object.
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

#Here we create matches of the descriptors, then we sort them based on their distances.
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(res1,kp1,res2,kp2,matches[:50],None, flags=6 | 6)
#plt.imshow(img3)
#img3= cv.cvtColor(img3,cv.COLOR_GRAY2BGR)

#plt.show()
plt.imsave('Matches_Stereo',img3)