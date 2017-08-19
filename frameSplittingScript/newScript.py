import cv2
import pandas as pd
import numpy as np
import pdb

'''
NOTE/TODO
Currently not doing any data cleaning can only
read in files that start with the actual data with no
junk above it 
 
'''

'''
Get video file from user use cv2 to: 
    1.) Write frames as jpgs in ./images/
    2.) Get the number of frames in the video (stored in count)
'''
pathToVideo = input("Enter Path to Video: ");
print("Exporting Frames from: ", pathToVideo)
print("\nFrames written: ")

vidcap = cv2.VideoCapture(pathToVideo)
count = 0

success, frame = vidcap.read()
while success:
  cv2.imwrite("frames/frame-%d.jpg" % count, frame)
  success, frame = vidcap.read()
  if count % 25 == 0:
    print(count)
  count += 1

print("Number of frames generated: ", count)
imageMatch = "frame-"+str(count)+".jpg" # TODO: what is imageMatch


'''
Status. 
    - count has the total number of frames
    - imageMatch is a mystery
    - continue to investigate this tmrw 
'''
pdb.set_trace()

# USER IO AND DATA CLEANING
telemetryInputPath = input("Enter Path to Telemetry Input File: ");
telemetryOutputPath = input("Enter Path to Telemetry Output File: ");                                                       
telemetryInputFile = open(telemetryInputPath, "r")
telemetryOutputFile = open(telemetryOutputPath, "w")
match = input("Enter the elapsed start time: "); # Why is this called match

# This might be useful later
#found = False
#while found != True: #Loop searches for the elapsed time you want to start export at
#    col = telemetryInputFile.readline().split()
#    if col[0] == match:
#        found = True


# Store telemetry input in a numpy array called telemetryInput
telemetryInput_pandas = pd.read_csv(telemetryInputPath, header=None)
telemetryInput = telemetryInput_pandas.values


# Compute the deltaTime array. Stores the elapsed time between each telemetry log entry.
# deltaTime[index] is the time difference between index and index+1
deltaTime = []
prevRow = []
for i, row in enumerate(telemetryInput):
    if i == 0:
        prevRow = row
        continue;

    dTime = row[0] - prevRow[0]
    deltaTime.append(dTime)
    prevRow = row
    
# Using the time between each log, compute how many frames were drawn in that time.
# Assign all of those frames in the range [currentTime, currentTime + deltaTime[log+1]]
# to the previous telemetry file. 
fps = 29.97 #TODO: change this to read from Dalton's code later.
framesPerBlock = []
for i, dTime in enumerate(deltaTime):
    frames = deltaTime[i] / (1/fps)  
    framesPerBlock.append(frames)

# Round frames to int. TODO. Must replace with roundingError.py when it is finished.
roundedFramesPerBlock = framesPerBlock[:]
for i in range(0, len(roundedFramesPerBlock)):
    roundedFramesPerBlock[i] = int(round(roundedFramesPerBlock[i], 0))

# need some of Dalton's code here:
image = ""
imageNum = "0"
imageType = ".jpg" # todo: this should be user input?
col = [0] * 4 # List with 4 0's 

j = 0
currentBlockNum = 0
for i,lines in enumerate(telemetryInputFile):    
    col[1] = telemetryInput[currentBlockNum][1]
    col[2] = telemetryInput[currentBlockNum][2]
    col[3] = telemetryInput[currentBlockNum][3]
    numFramesInCurrentBlock = roundedFramesPerBlock[currentBlockNum] # this doesn't need to be computed every time todo
    imageNum = str(imageNum)
    image = "frame-" + imageNum + imageType
    telemetryOutputFile.write(image + " {0} {1} {2}\n".format(str(col[1]), str(col[2]), str(col[3])))
    imageNum = int(imageNum) + 1
    j += 1

    if j == numFramesInCurrentBlock:
        currentBlockNum += 1
        j = 0


    '''
    Better way: delete image, imageMatch. Replace with: 
    
    if i+1 == count:    
    '''  
    if image == imageMatch:
        print("<Final Frame Reached>")
        print("<Telemetry Export Complete>")
        telemetryInputFile.close()
        telemetryOutputFile.close()
        quit()
