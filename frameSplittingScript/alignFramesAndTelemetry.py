# Use
getLogFileTimes()
getFrameTimes()
alignLogAndFrames()


# Return an array containing the time that each log file is logged
def getLogFileTimes(): 
    logFileTimes = []
    for row in telemetryInput:
        logFileTimes.append(row[0])

# Get an array that has a timestamp corresponding to each frame
# TODO: need fps
def getFrameTimes():
    fpsInverse = 1.0 / fps
    frameTimes = []
    for n in range(0, count):
        # Given frame n, the time it was taken is:  n * (1/fps)
        frameTimestamp = n * (fpsInverse)
        frameTimes.append(frameTimestamp)

# TODO: pass in correct arguments
'''
Input:
    - logFileTimes: Array of size # telemetry files - 1 (0 indexed). logFileTimes[i] gives the time the ith telemetry file was written.
    - frameTimes: Array of size # frames - 1 (0 indexed). frameTimes[i] gives the time the ith frame was taken from the camera. 
Output:
    - retval: An array of pairs (frame, logFileIdx). Returns the index of the logfile (logFileIdx) that each frame corresponds to.
'''
def alignLogAndFrames(logFileTimes, frameTimes):
    retval = []
    # TODO: check if the logfiles are smaller than 2
    # for each frame in the video
    prevLogFile = "0" # store index of prev log file
    for frame in range(0, count):
        time = frameTimes[frame]
        logFileIdx = getLogFileIdx(time, logFileTimes)

        retval.append((frame, logFileIdx))

    return retval

def getLogFileIdx(frameTime, logFileTimes):
    # compare with every frame?
    # compare with frame 0, 1, 2, 3
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

    
