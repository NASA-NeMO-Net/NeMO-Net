import pdb
# Use
#getLogFileTimes()
#getFrameTimes()
#alignLogAndFrames()


# Return an array containing the time that each log file is logged
# integrate: pass in telemetryInput
def getLogFileTimes(logfile): 
    logFileTimes = []
    for row in logfile:
        logFileTimes.append(row[0])
    return logFileTimes

# Not sure what time units the DJI phantom outputs the telemetry files in. Scale with fps timeline.
def scaleLogFileTimes(logfile, count, fps):
    # if we want to be super accurate we should also account for the frames before/after the
    # first/last log files.
    
    logFileTimes = logfile[:] # get copy since Python passes lists by ref
    time_start = logFileTimes[0]

    for i in range(0,len(logFileTimes)):
        logFileTimes[i] -= time_start
    return logFileTimes

# Get an array that has a timestamp corresponding to each frame
# integrate: need fps, count
def getFrameTimes(count, fps):
    fpsInverse = 1.0 / fps
    frameTimes = []
    for n in range(0, count):
        # Given frame n, the time it was taken is:  n * (1/fps)
        frameTimestamp = n * (fpsInverse)
        frameTimes.append(frameTimestamp)
    return frameTimes

# integrate: pass in getFrameTimes, getLogFileTimes
'''
Input:
    - logFileTimes: Array of size # telemetry files - 1 (0 indexed). logFileTimes[i] gives the time the ith telemetry file was written.
    - frameTimes: Array of size # frames - 1 (0 indexed). frameTimes[i] gives the time the ith frame was taken from the camera. 
Output:
    - retval: Returns a list s.t. retval[frame] = index of logfile of frame. 
'''
def alignLogAndFrames(frameTimes, logFileTimes, count):
    retval = []
    # TODO: check for edge cases
    for frame in range(0, count):
        time = frameTimes[frame]
        logFileIdx = getLogFileIdx(time, logFileTimes)
        retval.append(logFileIdx)
    return retval

def getLogFileIdx(frameTime, logFileTimes):
    # TODO: Optimize this algorithm
    minDistance = abs(frameTime - logFileTimes[0])
    logFileIdx = 0
    for i in range(1, len(logFileTimes)): # skip the first one since that's already in minDistance
        newDistance = abs(frameTime - logFileTimes[i])
        if newDistance < minDistance:
            minDistance = newDistance
            logFileIdx = i
        else: # If the distance increased, then we know we already found the minDistance. Break out of loop.
            break
    return logFileIdx


'''
# Can make a faster algorithm with these later (current one is O(n^2))
def getLeftLogFile():
    # keep going left until less than time
def getRightLogFile():
    # keep going right until more than time
''' 

    
