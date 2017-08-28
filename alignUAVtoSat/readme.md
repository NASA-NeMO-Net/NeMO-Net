### Installation
1. brew install gdal
2. Install QGIS (we need the osgeo library in your system's version of python (installation guide in qgisSetup.md))
3. Pip install numpy and pyproj on your system's version of python
4. Install OpenCV: http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  * Make sure to run "sudo make install" to also install on your system's version of python.

Note: The reason we're building everything on your system's version of python is because the osgeo library is difficult to get into a virtualenv (AFAIK).

### Usage
1. Find the UAV flight location in google earth. Write down the lat/long.
2. Jump to the lat/long in QGIS (make sure the current coordinate system is lat-lon/WGS84). The QGIS "Lat Lon Tools" Plugin is useful for this.
3. Zoom into the UAV area (making sure the current screen completely encapsulates the UAV flight path/spatial resolution. Export as tiff at the highest resolution ("Project -> Save as image...")
  * Currently, the screenshot must completely encapsulate the UAV path (this is assumed when linearly interpolating the UTM coordinates). Should probably change this later, but complicates the math.
4. Run the script and follow the steps. You'll likely have to blur the high res to be a similar resolution to the satellite imagery. I blurred with a gaussian kernel (available as blur.py); pyrDown is another option, but don't think downsampling is necessary here. View matched_features.jpg in the output folder to judge whether to blur more/less.
  * One step will ask for the UTM zone. You can find your UTM zone here: http://www.latlong.net/lat-long-utm.html
  * Make sure the world file (.tfw) is in the same directory as the google earth screenshot
  * You can double check that the .tiff image has lat/lon data by doing "gdalinfo [image]." Make sure the corner coordinates are in lat/lon, not pixels or UTM. If the coordinates are in pixels, you're probably missing the .tfw file. 
  * 75 is a good starting point for the number of point correspondences to use.Again, output/matched_features.jpg is a good image to look at to optimize this parameter. 
  * Can hit "Enter" to continue past OpenCV preview windows. 

### Output
The coordinates should be printed to console in UTM and lat/lon WGS84.
To overlay this image in QGIS, we need to create a worldfile (.tfw) for the raster image.

gdal_translate -of GTiff -a_srs EPSG:4326 -gcp 0 0 [upperLeft Lon] [upperLeft lat] -gcp 0 [height of image] [lowerLeft lon] [lowerLeft lat] -gcp [width of image] [height of image] [lowerRight lon] [lowerRight lat] uav_highres.tiff warped_uav_highres.tif

  * E.G gdal_translate -of GTiff -a_srs EPSG:4326 -gcp 0 0 -169.658707539 -14.1814672057 -gcp 0 4999 -169.657979775 -14.1821169937 -gcp 1104 4999 -169.657797838 -14.1819480485 uav_highres.tiff warped_uav_highres.tif

gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 warped_uav_highres.tif output.tif

listgeo -tfw  output.tif

  * Maybe try manually set extents of all 4 corners of the raster image and force write a worldfile using "listgeo -tfw  sift_bf_output_with_extents.tif"


### Debugging Bad Alignment
If you're getting bad results, try changing the number of point correspondences used. You can see the point correspondences in output/matched_features.jpg. Other images useful for debugging can be found in the output folder.

Note: Currently, the google earth screenshot must COMPLETELY encapsulate the UAV flightpath. This is beacuse completely encapsulating the UAV flightpath will result in no translation/rotation of the google earth screenshot during the homography, and this assumption is assumed when we linearly interpolate the UTM corner coordinates to find the corner coordinates of the UAV flight.

Note: We are currently writing points from SIFT to a txt file but not doing anything with it. If .txt file exists, should read from it and not run SIFT again. 