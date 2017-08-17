import sys
import numpy as np
import cv2
import cvk2
from osgeo import gdal,ogr,osr
from pyproj import Proj
import georefUtils
import pdb

print "Use: python ./sift_bf.py [path to blurred UAV] [path to google earth tiff w/ UTM georegistration] [# point correspondences] [# UTM zone (i.e. 2L for Ofu]"
if len(sys.argv) != 5:
    raise Exception("Incorrect Usage. See line above")
    
numPointCorrespondences = int(sys.argv[3])
utmZone = sys.argv[4]

# Read images
print "Reading images"
googleEarthPath = sys.argv[1]
googleEarth = cv2.imread(googleEarthPath,0)      
UAV = cv2.imread(sys.argv[2],0)
print "Done.\n"

# Feature detector
sift = cv2.SIFT() 

print "Finding image features"
# Detect keypoints in both images
(kp1,des1) = sift.detectAndCompute(googleEarth, None)
(kp2,des2) = sift.detectAndCompute(UAV, None)
print "Done.\n"

# Match point correspondences
print "Matching point correspondences"
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# matches = bf.match(des1,des2)
matches = bf.match(des1, des2)
print "Done.\n"

print "Sorting matches"
matches = sorted(matches, key=lambda val: val.distance)
print "Done.\n"

# Keep only the best numPointCorrespondences point pairs. 
goodPointCorrespondences = matches[:numPointCorrespondences]

# Draw matches visualization
out = georefUtils.drawMatches(googleEarth, kp1, UAV, kp2, goodPointCorrespondences)
cv2.imwrite("output/matched_features.jpg", out)

points_GE = open("output/points_GE.txt", "w")
points_UAV = open("output/points_UAV.txt", "w")

points_GE_all = open("output/points_GE_alll.txt", "w")
points_UAV_all = open("output/points_UAV_all.txt", "w")

_points_GE = []
_points_UAV = []

print "Memoizing features"
i = 0
for pair in matches:
    # Get the matching keypoints for each of the images
    googleEarth_idx = pair.queryIdx
    UAV_idx = pair.trainIdx

    # x - columns
    # y - rows
    (x1,y1) = kp1[googleEarth_idx].pt
    (x2,y2) = kp2[UAV_idx].pt
    
    if i < len(goodPointCorrespondences):
        points_GE.write("{0} {1}\n".format(x1, y1))
        points_UAV.write("{0} {1}\n".format(x2, y2))
        _points_GE.append((x1,y1))
        _points_UAV.append((x2,y2))
    points_GE_all.write("{0} {1}\n".format(x1, y1))
    points_UAV_all.write("{0} {1}\n".format(x2, y2))
    i += 1

points_GE.close()
points_UAV.close()
print "Done.\n"

print "Finding homography"
_points_GE = np.asarray(_points_GE)
_points_UAV = np.asarray(_points_UAV)

# Homography code modified from hw2 of Swarthmore ENGR 27. Includes starter code from Matthew Zucker. We warp googleEarth into the perspective of UAV

# Calculate image sizes
h_GE, w_GE = googleEarth.shape
size1 = (w_GE, h_GE)

h_UAV, w_UAV = UAV.shape
size2 = (w_UAV, h_UAV)

# Find a homography between the points
getHomography = cv2.findHomography(_points_UAV, _points_GE, method=cv2.RANSAC)

# Corner indices for each image
coords_GE = np.array( [ [[0, 0]],
                [[w_GE, 0]],
                [[w_GE, h_GE]],
                [[0, h_GE]] ], dtype='float32' )

coords_UAV = np.array([ [[0, 0]],
                     [[w_UAV, 0]],
                     [[w_UAV, h_UAV]],
                     [[0, h_UAV]] ], dtype='float32')

# Find homography mapping UAV to GE
coordsUAVtransformed = cv2.perspectiveTransform(coords_UAV, np.matrix(getHomography[0]))

# Find a rectangle that bounds both images (size out output image where both images will be overlaid)
pointsToEncapsulate = np.append(np.array(coords_GE), np.array(coordsUAVtransformed), axis=0)
boundingRectangle = cv2.boundingRect(pointsToEncapsulate)


# T is the translation to get rid of negative coordinates
T = np.array([ [1, 0, -1*boundingRectangle[0]],
    [0, 1, -1*boundingRectangle[1]],
    [0, 0, 1] ], dtype='float32')

# homographyPrime is the homography with the translation factor
homographyPrime = np.dot(T, np.matrix(getHomography[0]))

destSizeWH = (boundingRectangle[2], boundingRectangle[3])

# Calculate the perspective transform
warped_GE = cv2.warpPerspective(googleEarth, T, destSizeWH)
warped_UAV = cv2.warpPerspective(UAV, homographyPrime, destSizeWH)


# opencv window to preview tracked corners
uavCornersPreview = np.copy(warped_GE) # Note: this is a 3 channel RGB image. Only used for UI/preview. 
uavCornersPreview = cv2.cvtColor(uavCornersPreview, cv2.COLOR_GRAY2RGB) 
uavCornerIndices = []  # This is passed on to linearlyInterpolateUTM
uavCornerIndices.append(georefUtils.mapPointThroughHomography(homographyPrime, (0, 0)))
uavCornerIndices.append(georefUtils.mapPointThroughHomography(homographyPrime, (0, h_UAV)))
uavCornerIndices.append(georefUtils.mapPointThroughHomography(homographyPrime, (w_UAV, h_UAV)))
uavCornerIndices.append(georefUtils.mapPointThroughHomography(homographyPrime, (w_UAV, 0)))

pts = np.asarray(uavCornerIndices)
pts = pts.reshape((-1,1,2))
cv2.polylines(uavCornersPreview,[pts],True,(255,0,0))

# Show preview window
font = cv2.FONT_HERSHEY_SIMPLEX
for i, point in enumerate(uavCornerIndices):
    cv2.circle(uavCornersPreview, point, 3, 0)
    cv2.putText(uavCornersPreview, "corner"+str(i+1), point, font, 0.5, (0,0,0), 1)

cv2.imshow("Uav Corners Preview", uavCornersPreview)  # todo: make the image a bit more transparent
while cv2.waitKey(5) < 0: pass

cv2.imwrite("output/overlayP1.jpg", warped_GE)
cv2.imwrite("output/overlayP2.jpg", warped_UAV)

# Overlay the images. We simply add the images (leads to dark areas, but that makes for better visualization anyways)
finalImage = np.zeros_like(warped_GE)
finalImage = np.add(0.5*warped_GE, 0.5*warped_UAV)

print "Done.\n"

print "Writing output file \n"
cv2.imwrite("output/aligned_image.jpg", finalImage)
print "All done!\n"

##################################################################################################################
# Images are aligned. Now linearly interpolate UTM coordinates to find the corners of the UAV image
##################################################################################################################


'''
1.) Find the location on google earth. Note the lat/long.
2.) Jump to the lat/long in QGIS. Make sure the current coordinate system is lat/lon. Export as tiff w/ highest resolution. 
3.) Run script on this image and high res image (after BLURRING the high res to be a similar resolution)

'''
# Get the lat/lon corner coordinates of google earth screenshot 
gEarthScreenshot = gdal.Open(googleEarthPath, 0)
c = gEarthScreenshot.RasterXSize # width of img (# of cols)
r = gEarthScreenshot.RasterYSize # height of img (# of rows)

# todo: convert to pairs instead of depending on ordering
corners = georefUtils.getCorners(gEarthScreenshot, r, c)

# Needed because QGIS doens't support UTM zone 2L (Ofu/American Samoa). So we take in QGIS lat/lon data and covert it to UTM ourselves. 
cornersUTM = georefUtils.latlongListToUTM(corners, utmZone)

# We assume the google earth image was not translated (bad assumption, should change). 
# This does hold true is most cases though (if the UAV image is 100% encapsulated by the google earth image)
# todo: fix this assumption
indices = [] # Unfortunately for now, the order here matters. todo. 
indices.append((0, 0))
indices.append((h_GE, 0))
indices.append((w_GE, 0))
indices.append((h_GE, w_GE))
indices = np.asarray(indices)

uavCornersUTM = georefUtils.linearlyInterpolateUTM((h_GE,w_GE), cornersUTM, indices, (destSizeWH[1], destSizeWH[0]), uavCornerIndices)

finalPreview = np.copy(warped_GE) # Note: this is a 3 channel RGB image. Only used for UI/preview.
finalPreview = cv2.cvtColor(finalPreview, cv2.COLOR_GRAY2RGB)


# Connect the corners
cv2.polylines(finalPreview,[pts],True,(255,0,0))

# Not sure why this is necessary
uavCornersUTM = np.asarray(uavCornersUTM)

# show output as lat/lon
uavCornersLatLon = georefUtils.utmListToLatLong(uavCornersUTM, "2L")
uavCornersLatLon = np.asarray(uavCornersLatLon)

print "UAV Corners (UTM)" 
georefUtils.printCoordList(uavCornersUTM) 

uavCornersUTM = uavCornersLatLon

# at this point, uavCornersUTM and uavCornersLatLon are correct and in the right order

print "gEarth Corners (UTM)"
georefUtils.printCoordList(cornersUTM)

print "UAV Corners (lat/lon)"
georefUtils.printCoordList(uavCornersLatLon)

# Show labeled coordinates. 
font = cv2.FONT_HERSHEY_SIMPLEX
for i, point in enumerate(uavCornerIndices):
    cv2.circle(finalPreview, point, 3, 0)
    cv2.putText(finalPreview, "corner"+str(i+1) + ": ({0},{1})".format(round(uavCornersUTM[i][0],7), round(uavCornersUTM[i][1],7)), point, font, 0.5, (0,0,0), 1)
cv2.imshow("Overview", finalPreview)  # todo: make the image a bit more transparent
while cv2.waitKey(5) < 0: pass

# todo: encapsulate drawing chunks in functions? 
# todo: export UAV tiff with embedded Geo info. 
# todo: read from file if it exists
# todo: how many point correspondences to use? 
# todo: do we want to return the coordinates in lat/lon? 
# todo: get rid of order dependency
