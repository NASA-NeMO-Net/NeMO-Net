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
    2.) Get the number of frames in the video (stored in frameCount)
'''

fps = float(input("Input the UAV fps: "))

skip = input("Skip frame splitting? (y/n): ")
if skip == "n":
  pathToVideo = input("Enter Path to Video: ");
  print("Exporting Frames from: ", pathToVideo)
  print("\nFrames written: ")
  vidcap = cv2.VideoCapture(pathToVideo)
  frameCount = 0

  success, frame = vidcap.read()
  while success:
    cv2.imwrite("frames/frame-%d.jpg" % frameCount, frame)
    success, frame = vidcap.read()
    if frameCount % 25 == 0:
      print(frameCount)
    frameCount += 1

    print("Number of frames generated: ", frameCount)
else:
  frameCount = int(input("Enter the number of frames in the video: "))



'''
Status. 
    - frameCount has the total number of frames
    - continue to investigate this tmrw 
'''


# USER IO AND DATA CLEANING
telemetryInputPath = input("Enter Path to Telemetry Input File: ");
telemetryOutputPath = input("Enter Path to Telemetry Output File: ");

telemetryInputFile = open(telemetryInputPath, "r")
telemetryOutputFile = open(telemetryOutputPath, "w")

droneStartTime = input("Enter the elapsed drone start time: ");

# Store telemetry input in a numpy array called telemetryInput
telemetryInput_pandas = pd.read_csv(telemetryInputPath, header=None)
telemetryInput = telemetryInput_pandas.values

logFileTimes = newScript_utils.getLogFileTimes(telemetryInput)
logFileTimes = newScript_utils.scaleLogFileTimes(logFileTimes, frameCount, fps)
frameTimes = newScript_utils.getFrameTimes(frameCount, fps)
alignedPairs = newScript_utils.alignLogAndFrames(frameTimes, logFileTimes, frameCount)

image = ""
imageNum = "0"
imageType = ".jpg" # todo: this should be user input?
col = [0] * 4 # List with 4 0's 

currentBlockNum = 0
for i in range(0, frameCount):
    # pdb.set_trace()
    col[1] = telemetryInput[alignedPairs[i]][1]
    col[2] = telemetryInput[alignedPairs[i]][2]
    col[3] = telemetryInput[alignedPairs[i]][3]
    print("Aligning frame {0} with logfile: {1}".format(i, alignedPairs[i]))

    image = "frame-" + str(i) + imageType
    telemetryOutputFile.write(image + " {0} {1} {2}\n".format(str(col[1]), str(col[2]), str(col[3])))

    if i+1 == frameCount:
        print("<Final Frame Reached>")
        print("<Telemetry Export Complete>")
        telemetryOutputFile.flush()
        telemetryInputFile.close()
        telemetryOutputFile.close()
        quit()

print("All done!")
