#Dalton Kaua
#Written in Python 3.4.2 IDLE Shell
#For use readying telemetry for google earth mapping
print("<Starting .txt Export>")

front = "telemetry "
middle = "1"
end = " preprocessed.txt"
export = True
while export == True:
    name = front + str(middle) + end
    fileImport = open(name, "r");
    print("<Exporting...", name, ">")
    fileExport = open("Google "+front+str(middle)+".txt", "w")
    for lines in fileImport:
        line = lines
        col = lines.strip().split()
        fileExport.write(col[1])
        fileExport.write(" ")
        fileExport.write(col[2])
        fileExport.write("\n")
    middle = int(middle) + 1
    if middle == 52: #change if number of files differ
        export = False
        fileExport.close()
        fileImport.close()
        print("<File Export Complete>")
