

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



def getSeondsElapsed():
    # get the sum of dTimes of all previous frames


fps = 29.97
currentSecondsElapsed = 0
numberOfFramesDrawn = 0 # stores how many frames we have drawn
def getFrameCount():
    acutalSecondsElapsed = getSecondsElapsed()

    # can just use count
    f_secondsElapsed = # number of frames drawn * (1/fps) seconds per frame

    # The difference between if we estimated seconds elapsed based off of the
    # number of frames drawn vs the actual seconds ellapsed (based on our clock) 
    deltaTime = f_secondsElapsed - actualSecondsElapsed

    # if our deltaTime is positive then we are ahead of time and should round down.
    numFrames = int(round(roundedFramesPerBlock[i], 0)) # todo: this is currently wrong
    if deltaTime > 0:
        # round down numFrames
        numberOfFramesDrawn += numFrames
    else:
        # round up numFrames
        numberOfFrameDrawn += numFrames
    
    frameTarget = secondsElapsed * fps
    
    

    
    
'''
Todo: 
- getSecondsElapsed()
- f_secondsElapsed
- fix numFrames
'''




    
