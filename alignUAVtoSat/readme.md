# Guide

To run, python ./sift_bf.py [number of points to use in homography]
- e.g. python ./sift_bf.py 100

To try different images, change the imread lines (53,54)

The final outputted image is aligned_image.jpg
You can see the point correspondences in matched_features.jpg
pointsA.txt and pointsB.txt store the point correspondences, so we can read from file instead of recalculating the points every time (not yet implemented)

