import numpy as np
import cv2
from osgeo import gdal,ogr,osr
from pyproj import Proj
import pdb

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


# Get the coordinates of the image in UTM
def getCorners(gdalImage, r, c):
    gt = gdalImage.GetGeoTransform()
    xarr=[0,c]
    yarr=[0,r]

    retval = []
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            retval.append([y,x])        
    return retval

# Todo: pass in UTM zone (and return)
def latlongCoordinateToUTM(latlongCoord, UTM_zone):
    lat = latlongCoord[0]
    lon = latlongCoord[1]
    # myProj is thee function mapping WGS84 to UTM
    projOptions = "+proj=utm +zone={0}, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(UTM_zone)
    myProj = Proj(projOptions)

    # lon/lat to UTM
    # UTMy: Northing
    # UTMx: Easting
    UTMy, UTMx= myProj(lon, lat)
    
    # Reverse transformation is currently untested
    # UTM to lat/long
    # lat, lon = myProj(UTMx, UTMy, inverse=True)


    return np.asarray((UTMy, UTMx))

# Convert a list of latlong/Global Geodetic System to UTM
def latlongListToUTM(coords, UTM_zone):    
    retval = [] # List to store coordinates in UTM
    for coord in coords:
        UTMcoord = latlongCoordinateToUTM(coord, UTM_zone)
        retval.append(UTMcoord)
    return np.asarray(retval)


# Upper Left, Lower Left, Upper Right, Lower Right
# Assumes straight on google earth shot (no rotation)
def linearlyInterpolateUTM(size_GE, coords, indices, imgSize, UAV_indices): # indices in same order as coords, imgSize (r,c)--numpyarr
    # For now, we interpolate using the upper left corner and the lower right corner. 
    # TODO: Always store coordinates as a pair, so we don't have to work about sorting ruining everything

    # refactor coords as coords_GE and indices as indices_GE todo
    # Corner coordinates of googleEarth in UTM coordinates
    upperLeftUTM = coords[0]
    upperLeftIdx = indices[0]

    lowerLeftUTM = coords[1]
    lowerLeftIdx = indices[1]

    upperRightUTM = coords[2]
    upperRightIdx = indices[2]

    lowerRightUTM = coords[3]
    lowerRightIdx = indices[3]

    # Refer to picture
    # TODO: add picture

    dR = size_GE[0] # height of google earth screenshot
    dC = size_GE[1] # width of google earth screenshot

    # UTM distance between the two points (row component and col component) 
    dR_UTM = np.subtract(lowerLeftUTM, upperLeftUTM) # TODO: This should still be a pair
    dC_UTM = np.subtract(lowerRightUTM, lowerLeftUTM)

    dR_gradient = dR_UTM / dR
    dC_gradient = dC_UTM / dC

    # Use the gradient to linearly interpolate to the corners
    # Distance in form (r,c) (as in row distance to top left corner, col distance to top left corner)

    # Return UAV coordinates
    # Calculate UAV corner coordinates using this reference point
    referencePointUTM = upperLeftUTM
    referencePointIdx = upperLeftIdx

    # todo: check these indices
    uav0_idx = UAV_indices[0]
    uav1_idx = UAV_indices[1]
    uav2_idx = UAV_indices[2] 
    uav3_idx = UAV_indices[3]

    # maybe encapsulate all this into a function? 
    uav0_distance = uav0_idx - referencePointIdx
    uav0_UTM = referencePointUTM + np.dot(uav0_distance[0], dR_gradient) + np.dot(uav0_distance[1], dC_gradient)

    uav1_distance = uav1_idx - referencePointIdx
    uav1_UTM = referencePointUTM + np.dot(uav1_distance[0], dR_gradient) + np.dot(uav1_distance[1], dC_gradient)

    uav2_distance = uav2_idx - referencePointIdx
    uav2_UTM = referencePointUTM + np.dot(uav2_distance[0], dR_gradient) + np.dot(uav2_distance[1], dC_gradient)

    uav3_distance = uav3_idx - referencePointIdx
    uav3_UTM = referencePointUTM + np.dot(uav3_distance[0], dR_gradient) + np.dot(uav3_distance[1], dC_gradient)

    # Add to retval in the same order
    # Upper Left, Lower Left, Upper Right, Lower Right

    # TODO: also label image corners 
    # TODO: Convert to latlong again before returning
    retval = [] # for some reason we have to flip the coodinates here
    retval.append(uav3_UTM)
    retval.append(uav2_UTM)
    retval.append(uav1_UTM)
    retval.append(uav0_UTM)
    
    return retval

# Follow a point through the homography transform (point in (r,c) form)
def mapPointThroughHomography(homography, point):
    point_homog = (point[0], point[1], 1) # Write point in homogenous coordinates
    warped_point = np.matmul(homography, point_homog)
    warped_point = np.asarray(warped_point) # Not sure why this is necessary

    return (int(warped_point[0][0]), int(warped_point[0][1]))
    

def printCoordList(coords):
    print "\n"
    print "Warning: Coordinate prefixes/labels assume coordinate ordering in the list has not been changed and is the same as displayed by gdalinfo [imageName]"
    prefix = []
    prefix.append("Upper Left: ")
    prefix.append("Lower Left: ")
    prefix.append("Upper Right: ")
    prefix.append("Lower Right: ")
    for idx,coord in enumerate(coords):
        print "{0} ({1}, {2}) \n".format(prefix[idx],round(coord[0],3), round(coord[1],3))
    
'''
tiff = "latlong.tiff"
img = gdal.Open(tiff)
c = img.RasterXSize
r = img.RasterYSize
corners  = getCorners(img, r, c)
cornersUTM = latlongListToUTM(corners, "2L")
printCoordList(corners)
printCoordList(cornersUTM)
'''



# Check lat, lon ordering
# Interpolate

# Potentially convert back to lat lon
# Embed in tiff geo info and load in qgis
