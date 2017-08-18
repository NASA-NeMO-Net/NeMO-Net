#Dalton Kaua
#Jake Burger
#Written in Python 3.4.2 IDLE Shell
#FluidCam Telemetry Data Processing
#Orig Ver Available on Google Drive
print("<Telemetry Export Script>")
print("<Ensure that this script is placed in the same folder as all materials used>")
print("<Include file types like .txt and .mov when the script asks for a name     >")
#------------------------------------------------
#OpenCV code block
fileVideo = input("Enter Name of Video: ");

#use fileVideo to open the video and run OpenCV functions to
#seperate the video into frames using a specific path to save them in
#a different folder. Be sure to use the naming convention
#frame-000000.jpg and increment the number. Open folder to see the last frame's name and
#continue with the rest of the code.






#------------------------------------------------
fileInput = input("Enter Name of Telemetry Input: ");
fileOutput = input("Enter Name of Telemetry Output: ");
fileIn = open(fileInput, "r")
fileOut = open(fileOutput, "w")
match = input("Enter the elapsed start time: ");
imageMatch = input("Enter name of end frame: ");#Name of final frame exported from OpenCV
line = ""
found = False
while found != True: #Loop searches for the elapsed time you want to start export at
    col = fileIn.readline().split()
    if col[0] == match:
        found = True
print("<Start Time Found>")
print ("<Starting Read>")
image = ""
imageName = "frame-"
imageNum = "0"
imageType = ".jpg"
print ("<Reading...>")
for lines in fileIn:
    j = 0
    line = lines
    col = line.strip().split()
    while j != 3: #3 = 30Hz <- change depending on vid
        imageNum = str(imageNum)
        image = imageName + imageNum.zfill(6) + imageType
        fileOut.write(image)
        fileOut.write(" ")
        fileOut.write(col[1])
        fileOut.write(" ")
        fileOut.write(col[2])
        fileOut.write(" ")
        fileOut.write(col[3])
        fileOut.write("\n")
        imageNum = int(imageNum) + 1      
        j = j + 1
        if image == imageMatch:
            print("<Final Frame Reached>")
            print("<Telemetry Export Complete>")
            fileIn.close()
            fileOut.close()
            quit()
print("ERR: Final Frame Not Reached")
print("Possible Causes:")
print("-Incorrect Telemetry File")
print("->Solution: load the telemetry file and video start coordinates into google earth pro to ensure they are on the same path/track")
print("-Incorrect Ratio Number")
print("->Solution: Some videos may not be recorded at 30fps or 30Hz. Manually change the ratio in the code to match video frame rate")
fileIn.close()
fileOut.close()
#Step 2
