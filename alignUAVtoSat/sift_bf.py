import sys
import numpy as np
import cv2
import cvk2
from matplotlib import pyplot as plt
import pdb

if len(sys.argv) == 1:
    raise Exception("Please pass a commandline argument for the number of point correspondences to use")
    
numPointCorrespondences = int(sys.argv[1])

# method taken from rayryeng, StackOverflow
def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Also return the image if you'd like a copy
    return out

print "Reading images"
img1 = cv2.imread('images/transect_highres_gEarth.jpg',0)      
img2 = cv2.imread('images/transect_lowres_blurred.tiff',0)

# Write black and white images
#cv2.imwrite("img1.jpg", img1) 
#cv2.imwrite("img2.jpg", img2)

print "Done.\n"

# Feature detector
sift = cv2.SIFT() 

print "Finding image features"
# Detect keypoints in both images
(kp1,des1) = sift.detectAndCompute(img1, None)
(kp2,des2) = sift.detectAndCompute(img2, None)
print "Done.\n"


print "Matching point correspondences"
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1,des2)
print "Done.\n"

print "Sorting matches"
matches = sorted(matches, key=lambda val: val.distance)
print "Done.\n"

# Keep only the best numPointCorrespondences point pairs. 
goodPointCorrespondences = matches[:numPointCorrespondences]

# Draw matches visualization
out = drawMatches(img1, kp1, img2, kp2, goodPointCorrespondences)
cv2.imwrite("matched_features.jpg", out)

# change pointsA/pointsB to filename later
pointsA = open("pointsA.txt", "w")
pointsB = open("pointsB.txt", "w")

pointsA_all = open("pointsA_all.txt", "w")
pointsB_all = open("pointsB_all.txt", "w")

_pointsA = []
_pointsB = []

# Find point correspondences and put in txt file for hw2_starter.py
print "Memorizing features"
i = 0
for pair in matches:
    # Get the matching keypoints for each of the images
    img1_idx = pair.queryIdx
    img2_idx = pair.trainIdx

    # x - columns
    # y - rows
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    
    if i < len(goodPointCorrespondences):
        pointsA.write("{0} {1}\n".format(x1, y1))
        pointsB.write("{0} {1}\n".format(x2, y2))
        _pointsA.append((x1,y1))
        _pointsB.append((x2,y2))
    pointsA_all.write("{0} {1}\n".format(x1, y1))
    pointsB_all.write("{0} {1}\n".format(x2, y2))
    i += 1

pointsA.close()
pointsB.close()
print "Done.\n"

print "Finding homography"
_pointsA = np.asarray(_pointsA)
_pointsB = np.asarray(_pointsB)

# Homography code modified from hw2 of Swarthmore ENGR 27. Includes starter code from Matthew Zucker. We warp img1 into the perspective of img2

#Calculate image sizes
h1, w1 = img1.shape
size1 = (w1, h1)

h2, w2 = img2.shape
size2 = (w2, h2)

# Find a homography between the points
getHomography = cv2.findHomography(_pointsA, _pointsB, method=cv2.RANSAC)

#coords1 array of original coordinates (image 1)
#corner points of B
coords1 = np.array( [ [[0, 0]],
                [[w1, 0]],
                [[w1, h1]],
                [[0, h1]] ], dtype='float32' )

#coords to transform (image 2 coordinates in the coordinate system of image 1)
coords2 = np.array([ [[0, 0]],
                     [[w2, 0]],
                     [[w2, h2]],
                     [[0, h2]] ], dtype='float32')

# Find homography mapping original to target (img2 to img1)
coords2transformed = cv2.perspectiveTransform(coords2, np.matrix(getHomography[0]))

# Find a rectangle that bounds both images (size out output image where both images will be overlaid)
pointsToEncapsulate = np.append(np.array(coords1), np.array(coords2transformed), axis=0)
# Bounding rect is not working correctly, maybe implement by hand (min/max of each all x/y)
boundingRectangle = cv2.boundingRect(pointsToEncapsulate)

#STEP 4: CALCULATE H'
#Add translation to img1 to fit in bounding rectangle
T = np.array([ [1, 0, -1*boundingRectangle[0]],
    [0, 1, -1*boundingRectangle[1]],
    [0, 0, 1] ], dtype='float32')

homographyPrime = np.dot(T, np.matrix(getHomography[0]))

#STEP 5: Warp A with H' into size (Wc, Hc)
destSizeWH = (boundingRectangle[2], boundingRectangle[3])

# Warp images using homography 
warpedA = cv2.warpPerspective(img1, homographyPrime, destSizeWH)
warpedB = cv2.warpPerspective(img2, T, destSizeWH) # Just a translation

#todo: warp 4 corner coordinates

#todo: interpolate cartesian coordinates across the entire image to georeference the entire img
# Do we need to fully georectify the image or just mark the corners (and embed in tiff?)

#something is wrong with this warpPerspective
cv2.imwrite("overlayP1.jpg", warpedA)
cv2.imwrite("overlayP2.jpg", warpedB)

#LAST STEP: Overlay the images
finalImage = np.zeros_like(warpedA)
#simply add the images, leads to dark areas
finalImage = np.add(0.5*warpedA, 0.5*warpedB)


# verify coordinate systems + homography mapping is valid

print "Done.\n"

print "Writing output file \n"
cv2.imwrite("aligned_image.jpg", finalImage)
print "All done!\n"












