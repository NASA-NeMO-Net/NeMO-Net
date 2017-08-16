1. Find the location on google earth. Write down the lat/long.
2. Jump to the lat/long in QGIS (make sure the current coordinate system is lat/lon). The QGIS "Lat Lon Tools" Plugin is useful for this.
3. Zoom into the UAV area (making sure the current screen completely encapsulates the UAV flight path/spatial resolution. Export as tiff at the highest resolution ("Save as image...")
-- Currently, the screenshot must completely encapsulate the UAV path (this is assumed when linearly interpolating the UTM coordinates). Should probably change this later. 
4. Run script on this image and high res image (after BLURRING the high res to be a similar resolution). I blurred with a gaussian kernel (blur.py); pyrDown is another option, but don't think downsampling is necessary here.  
-- Look at run.sh for an example. You can find your UTM zone here: http://www.latlong.net/lat-long-utm.html
-- python ./sift_bf.py [gEarth/qgis screenshot w/ lat/lon meta data] [blurred/downsampled UAV image] [max # of point correspondences] [UTM zone of UAV] 


NOTE: images/transect_lowres_blurred.tif is in the gdrive (too big for git)