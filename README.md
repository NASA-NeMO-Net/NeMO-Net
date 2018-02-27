# NeMO-NET
Neural Multi-Modal Observation & Training Network for Global Coral Reef Assessment
Created by: Alan Li
Email: alan.s.li@nasa.gov

Startup notes:
Installation notes:
  - Install OpenCV. Instructions for full mac install here: http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  - Make sure the following Python packages are installed: numpy, matplotlib, jupyter, tensorflow, PIL, gdal
  - Install CUDA and GPU Acceleration (version 8 preferable) for NVIDIA cards, if applicable (CUDA 9 has been verified to work with updated Tensorflow v 1.5, with a few deprecation warnings)
  - Install Keras version 2.0.8
  - Install hyperopt and hyperas (pip install)
  - Install pydensecrf, if fully connected CRFs are used (requires cython and rc.exe + rcdll.dll from C:\Program Files (x86)\Windows Kits\8.0\bin\x86 to C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin)
  - Downgrade networkx to 1.11 (pip install networkx==1.11)
  - Additional resources: https://www.lfd.uci.edu/~gohlke/pythonlibs for .whl files

3) Images can be downloaded per request, and are to be contained within the ./Images/ folder
