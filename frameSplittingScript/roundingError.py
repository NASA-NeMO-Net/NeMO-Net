

'''
Video: 29.97 fps
Telemetry file: 

Time between each GPS checkin is... variable? 
    - call it dTime

deltaTime: array that holds dtimes. deltaTime[index] is the dTime b/w index, index+1


29.97 frames/second -> 1/29.97 seconds/frame
So dTime/(1/fps) gives us the number of frames in dTime

NOTE: it will be a decimal. 



Frame target is: seconds elapsed * 29.97fps


'''

fps = 29.97
currentSecondsElapsed = 0
numberOfFramesDrawn = 0 # stores how many frames we have drawn

def getSeondsElapsed():
    # get the sum of dTimes of all previous frames
    return int(count * fps)

f_secondsElapsed = 0
def getFrameCount():
    acutalSecondsElapsed = getSecondsElapsed() # "Frame target" 

    # The difference between if we estimated seconds elapsed based off of the
    # number of frames drawn vs the actual seconds ellapsed (based on our clock) 
    deltaTime = f_secondsElapsed - actualSecondsElapsed
    
    # if our deltaTime is positive then we are ahead of time and should round down.
    if deltaTime > 0:
        # round down numFrames
        numFrames = 
        numberOfFramesDrawn += numFrames
    elif deltaTime < 0:
        # round up numFrames
        numberOfFramesDrawn += numFrames
    else:
        numberOfFramesDrawn += int(round(roundedFramesPerBlock[i], 0))
    f_secondsElapsed = # number of frames drawn * (1/fps) seconds per frame

'''
Another way:
Split based of log entry times. Just calculate *when* they were entered. 
Should also probably classify each frame to the closest log file, not the previous one

But have to figure out how time works in the logfile
[  |         |   |      |          |     |         | ]
[   print telemetry file corresponding to interval   ]

'''


    
'''
Todo: 
- getSecondsElapsed()
- f_secondsElapsed
- fix numFrames
'''




    
