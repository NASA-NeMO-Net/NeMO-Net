# Installing QGIS
Follow the instructions here: https://arset.gsfc.nasa.gov/sites/default/files/land/webinars/Advanced_Land_Classification/ARSET_Downloading_Installing_QGIS_Final.pdf

# Setting Up Your Plugin Builder Environment
1. Get the QGIS Plugin Builder. Plugins -> Manage and Install Plugins -> Search "Plugin Builder" and install.
2. Show experimental plugins in your plugin manager. Plugins -> Manage and Install Plugins -> Settings -> Check "Show also experimental plugins".
3. Get the QGIS Plugin Reloader. Plugins -> Manage and Install Plugins -> Search "Plugin Reloader" and install.
4. Use the Plugin Builder. Plugins -> Plugin Builder -> Plugin Builder. Complete the dialog that comes up, making sure to remember what you put for "plugin class" and "plugin name." Save in /Users/[username]/.qgis2/python/plugins. 
  * NOTE: to access .qgis2 on OSC in the finder save dialog, use "command + shift + ." to show all files (i.e. dotfiles)
5. Download pyrcc4 tools (brew install pyqt doesn't work. As of Aug 2017, use "brew install cartr/qt4/pyqt" to get the correct version of pyqt <--- very important
6. Run the makefile in your plugin directory (e.g. cd /Users/[username]/.qgis2/python/plugins/[plugin name], then "make")
7. Go to plugins -> Manage and Install Plugins -> Search [your plugin name] -> Hit the checkbox next to your plugin name  (might have to restart QGIS)
8. All the heavy lifting will be done in test_class.py in 
/Users/[username]/.qgis2/python/plugins/[plugin class name]/[plugin module name]

# Creating a Plugin
Good tutorial: https://www.youtube.com/watch?v=ZI3yzAVCK_0
Summary:
  * Use QT Creator to create the UI Interface
  * Everything starts in /Users/[username]/.qgis2/python/plugins/[plugin class name]/[plugin module name]/[module_name].py in the run() method.

