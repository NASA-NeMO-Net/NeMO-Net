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
fileInput = input("Enter Name of Telemetry Input: ");                                               fileOutput = input("Enter Name of Telemetry Output: ");                                                       
fileIn = open(fileInput, "r")
fileOut = open(fileOutput, "w")
match = input("Enter the elapsed start time: ");                                                              
line = ""
#found = False
#while found != True: #Loop searches for the elapsed time you want to start export at
#    col = fileIn.readline().split()
#    if col[0] == match:
#        found = True


# CALCULATE A BLOCKS
data = pd.read_csv(fileInput, header=None)
data = data.values
temp = round((data[1][0]-data[0][0])/0.03337,1)
print(temp)

deltaTime = []

prevRow = []
for i, row in enumerate(data):
    if i == 0:
        prevRow = row
        continue;

    dTime = row[0] - prevRow[0]
    # a_value = int(round(a_value, 1))
    deltaTime.append(dTime)
    prevRow = row
    

# deltaTime[index] is the time difference between index and index+1

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
imageName = "frame-"
imageNum = "0"
imageType = ".jpg"
col = [0] * 4 # List with 4 0's 

j = 0
currentBlockNum = 0
for i,lines in enumerate(fileIn):    
    col[1] = data[currentBlockNum][1]
    col[2] = data[currentBlockNum][2]
    col[3] = data[currentBlockNum][3]
    numFramesInCurrentBlock = roundedFramesPerBlock[currentBlockNum]
    imageNum = str(imageNum)
    image = imageName + imageNum + imageType
    fileOut.write(image + " {0} {1} {2}\n".format(str(col[1]), str(col[2]), str(col[3])))
    imageNum = int(imageNum) + 1
    j += 1

    if j == numFramesInCurrentBlock:
        currentBlockNum += 1
        j = 0
            
    if image == imageMatch:
        print("<Final Frame Reached>")
        print("<Telemetry Export Complete>")
        fileIn.close()
        fileOut.close()
        quit()
