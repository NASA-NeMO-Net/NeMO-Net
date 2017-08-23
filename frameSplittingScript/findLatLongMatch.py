from math import cos, asin, sqrt
import Queue
import glob, os
import pdb

# Returns km distance between two lat/lon point using the optimized Haversine formula
# https://en.wikipedia.org/wiki/Haversine_formula
def getDistance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) #2*R*asin...


print("Enter the path to your folder containing PREPROCESSED telemetry files.")
print("current directory: use '.'")
print("folder in current dir: use './[folderName]/' \n")
telemetryFolder = raw_input("Path: ")
os.chdir(telemetryFolder)

targetLat = input("Enter Lat: ")
targetLon = input("Enter Long: ")


#topFive = Queue.PriorityQueue()

topFive = [""] * 5

minDistances = []
for fileName in glob.glob("*.txt"):
    telemetryFile = open(fileName, "r")
    for idx,line in telemetryFile:
        lat = line[1]
        lon = line[2]
        distance = getDistance(targetLat, targetLon, lat, lon)

        temp = {}
        temp["distance"] = distance
        temp["lat"] = lat
        temp["lon"] = lon
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

print "Top Five matches"
print topFive
    
        
        


