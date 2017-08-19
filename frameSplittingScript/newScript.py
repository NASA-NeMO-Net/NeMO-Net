import cv2
import pandas as pd
import numpy as np
import pdb
import newScript_utils

'''
NOTE/TODO
Currently not doing any data cleaning can only
read in files that start with the actual data with no
junk above it 
 
'''

'''
Get video file from user use cv2 to: 
    1.) Write frames as jpgs in ./frames
    2.) Get the number of frames in the video (stored in count)
'''

fps = 29.97 # TODO: take this as user input
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

logFileTimes = newScript_utils.getLogFileTimes(telemetryInput)
logFileTimes = newScript_utils.scaleLogFileTimes(logFileTimes, count, fps)
frameTimes = newScript_utils.getFrameTimes(count, fps)
alignedPairs = newScript_utils.alignLogAndFrames(frameTimes, logFileTimes, count)

# need some of Dalton's code here:
image = ""
imageNum = "0"
imageType = ".jpg" # todo: this should be user input?
col = [0] * 4 # List with 4 0's 

j = 0
currentBlockNum = 0
for i,lines in enumerate(telemetryInputFile):
    col[1] = telemetryInput[alignedPairs[i]][1]
    col[2] = telemetryInput[alignedPairs[i]][2]
    col[3] = telemetryInput[alignedPairs[i]][3]
    print(i)

    imageNum = str(imageNum)
    image = "frame-" + imageNum + imageType
    telemetryOutputFile.write(image + " {0} {1} {2}\n".format(str(col[1]), str(col[2]), str(col[3])))
    imageNum = int(imageNum) + 1
    j += 1


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
