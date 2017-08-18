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
# OPEN CV
fileVideo = input("Enter Name of Video: ");                                                                   
print("Exporting Frames from: ", fileVideo)
vidcap = cv2.VideoCapture(fileVideo)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  #count = str(count)
  cv2.imwrite("images/frame-%d.jpg" % count, image)
  count = int(count)
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1
count = count - 2
print("Number of frames generated: ", count)
imageMatch = "frame-"+str(count)+".jpg"


# USER IO AND DATA CLEANING
fileInput = input("Enter Name of Telemetry Input: ");                                                         
fileOutput = input("Enter Name of Telemetry Output: ");                                                       
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

fps = 29.97 # change this to read from Dalton's code later
framesPerBlock = []
for i, dTime in enumerate(deltaTime):
    frames = deltaTime[i] / (1/fps)  
    framesPerBlock.append(frames)

roundedFramesPerBlock = framesPerBlock[:]

for i in range(0, len(roundedFramesPerBlock)):
    roundedFramesPerBlock[i] = int(round(roundedFramesPerBlock[i], 0))





# need some of Dalton's code here:
image = ""
imageName = "frame-"
imageNum = "0"
imageType = ".jpg"
col = []
col.append(0) # todo change this
col.append(0)
col.append(0)
col.append(0)


j = 0
currentBlockNum = 0
for i,lines in enumerate(fileIn):

    #col = line.strip().split()
    
    col[1] = data[currentBlockNum][1]
    col[2] = data[currentBlockNum][2]
    col[3] = data[currentBlockNum][3]
    numFramesInCurrentBlock = roundedFramesPerBlock[currentBlockNum]
    #while j <= numFramesInCurrentBlock: 
    imageNum = str(imageNum)
    image = imageName + imageNum + imageType
    fileOut.write(image)
    fileOut.write(" ")
    fileOut.write(str(col[1]))
    fileOut.write(" ")
    fileOut.write(str(col[2]))
    fileOut.write(" ")
    fileOut.write(str(col[3]))
    fileOut.write("\n")
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
