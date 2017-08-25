from math import cos, asin, sqrt
from multiprocessing import Queue
import glob, os
import pdb

def getDistance(targetLat, targetLon, lat, lon):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat - targetLat) * p)/2 + cos(targetLat * p) * cos(lat * p) * (1 - cos((lon - targetLon) * p)) / 2
    return 1000 * 12742 * asin(sqrt(a)) #2*R*asin...   added *1000 to return meters, not km

print ("Have all telemetry files as .txt and in the same folder")

telemetryFolder = input("Path to folder with all Telemetry Files: ") #raw_input
os.chdir(telemetryFolder)

targetLat = input("Given Lat (decimal) from Video: ")
targetLon = input("Given Long (decimal) from Video: ")
targetLat = float(targetLat)
targetLon = float(targetLon)


#topFive = Queue.PriorityQueue()

topFive = [""] * 5

for fileName in glob.glob("*.csv"):
    print (fileName)

minDistances = []
for fileName in glob.glob("*.txt"):
    telemetryFile = open(fileName, "r")
    for line in telemetryFile:
        line = line.strip().split()            

        lat = float(line[1])
        lon = float(line[2])
        distance = getDistance(targetLat, targetLon, lat, lon)   
        time = float(line[0])
    
        temp = {}
        temp["distance"] = distance
        temp["lat"] = lat
        temp["lon"] = lon
        temp["time"] = time
        temp["telemetryFile"] = fileName
        if topFive[0] == "" or distance < topFive[0]["distance"]:
            topFive[0] = temp
            continue
        elif topFive[1] == "" or distance < topFive[1]["distance"]:
            topFive[1] = temp
            continue
        elif topFive[2] == "" or distance < topFive[2]["distance"]:
            topFive[2] = temp
            continue
        elif topFive[3] == "" or distance < topFive[3]["distance"]:
            topFive[3] = temp
            continue
        elif topFive[4] == "" or distance < topFive[4]["distance"]:
            topFive[4] = temp
            continue                   
    telemetryFile.close()
print ("Our target coordinate is ({0},{1})".format(targetLat, targetLon))
for match in topFive:
    if match["distance"] == 0:
        print ("PERFECT MATCH: "),
    print ("({0},{1}) is {2} meters away from our target coordinate: Time = {3} in {4}".format(match["lat"], match["lon"], match["distance"], match["time"], match["telemetryFile"]))
