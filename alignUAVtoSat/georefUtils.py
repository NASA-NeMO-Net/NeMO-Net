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

    temp = []
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            temp.append([y,x])
    retval = {}
    retval["upperLeft"] = temp[0]
    retval["lowerLeft"] = temp[1]
    retval["upperRight"] = temp[2]
    retval["lowerRight"] = temp[3]

    return retval

'''
Input:
    utmCoord: utmCoordinate we want to convert (Easting, Northing). Numpy array or python list. 
    UTM_zone: string containing UTM zone we're working in (e.g. "2L" for Ofu)
Output: 
    retval: (lat, lon) pair. 
'''
def utmCoordinateToLatLong(utmCoord, UTM_zone):
    UTMx = utmCoord[0]
    UTMy = utmCoord[1] 
    # myProj is thee function mapping WGS84 to UTM
    projOptions = "+proj=utm +zone={0}, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(UTM_zone)
    myProj = Proj(projOptions)
    
    # Reverse transformation is currently untested
    # UTM to lat/long
    lon, lat = myProj(UTMx, UTMy, inverse=True)
    
    retval = (lat,lon)
    return retval

# todo: what does coords look like? need to convert to 
# todo: does this preserve order? 
def utmDictionaryToLatLong(coords, UTM_zone):
    retval = {}
    retval["upperLeft"] = utmCoordinateToLatLong(coords["upperLeft"], UTM_zone)
    retval["lowerLeft"] = utmCoordinateToLatLong(coords["lowerLeft"], UTM_zone)
    retval["lowerRight"] = utmCoordinateToLatLong(coords["lowerRight"], UTM_zone)
    retval["upperRight"] = utmCoordinateToLatLong(coords["upperRight"], UTM_zone)

    return retval

def latLongCoordinateToUtm(latlongCoord, UTM_zone):
    lat = latlongCoord[0]
    lon = latlongCoord[1]
    # myProj is thee function mapping WGS84 to UTM
    projOptions = "+proj=utm +zone={0}, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(UTM_zone)
    myProj = Proj(projOptions)

    # lon/lat to UTM
    # UTMy: Northing
    # UTMx: Easting
    UTMy, UTMx= myProj(lon, lat)
    
    return np.asarray((UTMy, UTMx))

# Convert a list of latlong/Global Geodetic System to UTM
def latLongDictionaryToUtm(coords, UTM_zone):    
    retval = {} # List to store coordinates in UTM
    retval["upperLeft"] = latLongCoordinateToUtm(coords["upperLeft"], UTM_zone)
    retval["lowerLeft"] = latLongCoordinateToUtm(coords["lowerLeft"], UTM_zone)
    retval["lowerRight"] = latLongCoordinateToUtm(coords["lowerRight"], UTM_zone)
    retval["upperRight"] = latLongCoordinateToUtm(coords["upperRight"], UTM_zone)

    return retval


# Upper Left, Lower Left, Upper Right, Lower Right
# Assumes straight on google earth shot (no rotation)
def linearlyInterpolateUTM(size_GE, coords, indices, imgSize, UAV_indices): # indices in same order as coords, imgSize (r,c)--numpyarr
    # For now, we interpolate using the upper left corner and the lower right corner. 
    # TODO: Always store coordinates as a pair, so we don't have to work about sorting ruining everything

    # refactor coords as coords_GE and indices as indices_GE todo
    # Corner coordinates of googleEarth in UTM coordinates
    upperLeftUTM = coords["upperLeft"]
    upperLeftIdx = indices["upperLeft"]

    lowerLeftUTM = coords["lowerLeft"]
    lowerLeftIdx = indices["lowerLeft"]

    lowerRightUTM = coords["lowerRight"]
    lowerRightIdx = indices["lowerRight"]

    upperRightUTM = coords["upperRight"]
    upperRightIdx = indices["upperRight"]

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
    uav_upperLeft_idx = UAV_indices["upperLeft"] # HAVE BEEN FLIPPED TODO
    uav_lowerLeft_idx = UAV_indices["lowerLeft"]
    uav_lowerRight_idx = UAV_indices["lowerRight"] 
    uav_upperRight_idx = UAV_indices["upperRight"]

    # maybe encapsulate all this into a function? 
    # todo: flipped indices
    # tood: drawing in wrong order in preview window
    uav_upperLeft_distance = uav_upperLeft_idx - referencePointIdx
    uav_upperLeft_UTM = referencePointUTM + uav_upperLeft_distance[1]*dR_gradient + uav_upperLeft_distance[0]*dC_gradient
    
    uav_lowerLeft_distance = uav_lowerLeft_idx - referencePointIdx
    uav_lowerLeft_UTM = referencePointUTM + uav_lowerLeft_distance[1]*dR_gradient + uav_lowerLeft_distance[0]*dC_gradient

    # the problem
    uav_lowerRight_distance = uav_lowerRight_idx - referencePointIdx
    uav_lowerRight_UTM = referencePointUTM + uav_lowerRight_distance[1]*dR_gradient + uav_lowerRight_distance[0]*dC_gradient

    uav_upperRight_distance = uav_upperRight_idx - referencePointIdx
    uav_upperRight_UTM = referencePointUTM + uav_upperRight_distance[1]*dR_gradient + uav_upperRight_distance[0]*dC_gradient

    retval = {}
    retval["upperLeft"] = uav_upperLeft_UTM
    retval["lowerLeft"] = uav_lowerLeft_UTM
    retval["lowerRight"] = uav_lowerRight_UTM
    retval["upperRight"] = uav_upperRight_UTM

    return retval

# Follow a point through the homography transform (point in (r,c) form)
def mapPointThroughHomography(homography, point):
    point_homog = (point[0], point[1], 1) # Write point in homogenous coordinates
    warped_point = np.matmul(homography, point_homog)
    warped_point = np.asarray(warped_point) # Not sure why this is necessary

    return (int(warped_point[0][0]), int(warped_point[0][1]))

'''
Input: 
    - dict: Takes a dictionary holding the for corners of an image. The corners should
      be named upperLeft, lowerLeft, lowerRight, and upperRight
Output:
    - retval: np array containing the corners in the order upperLeft, lowerLeft, 
      lowerRight, and upperRight
'''
def cornersDictToNumpyArray(dict):
    retval = []
    retval.append(dict["upperLeft"])
    retval.append(dict["lowerLeft"])
    retval.append(dict["lowerRight"])
    retval.append(dict["upperRight"])

    retval = np.asarray(retval)
    return retval

def printCoordDictionary(coords):
    print("upperLeft: {0}".format(coords["upperLeft"]))
    print("lowerLeft: {0}".format(coords["lowerLeft"]))
    print("lowerRight: {0}".format(coords["lowerRight"]))
    print("upperRight: {0}\n".format(coords["upperRight"]))
        
