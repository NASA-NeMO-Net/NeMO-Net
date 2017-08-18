#Dalton Kaua
#Written in Python 3.4.2 IDLE Shell
#FluidCam Telemetry Data Processing
print("<Coordinate Search Script>")
print("WARNING: Searches for exact coordinates")
print("Upon completion select cancel")
Lat = input("Enter Lat: ");
Long = input("Enter Long: ");
print("<Searching for Location...>")
front = "telemetry " #Change depending on the name of the telemetry files
middle = "1"         #Incr
end = " preprocessed.txt"
line = ""
search = True
start = True
found = False
while search == True:
    name = front + str(middle) + end
    fileSearch = open(name, "r");
    print("<Searching", name, ">")
    if start == True:
        lines = ""
        start = False
    for lines in fileSearch:
        line = lines
        col = lines.strip().split()
        if col[1] == Lat:
            print("Latitude Match at Elapsed Time: ", col[0])
        if col[2] == Long:
            print("Longitude Match at Elapsed Time: ", col[0])
        if col[1] == Lat:
           if col[2] == Long:
                print("Full Match Found at Elapsed Time:", col[0], "<----------")
                found = True
    middle = int(middle) + 1
    if int(middle) == 52: #change if different number of telemetry files
        fileSearch.close()
        if found != True:
            print("ERR: No Match Found!")
        else:
            print("<Search Complete>")
        quit()
        
#Step 1
