# NeMO-NET
Neural Multi-Modal Observation & Training Network for Global Coral Reef Assessment
Created by: Alan Li
Team: Ved Chirayath, Michal Segal-Rozenhaimer, Juan L. Torress-Peres, Jarrett van den Bergh
Website: http://nemonet.info/
Email: alan.s.li@nasa.gov

The installation is meant to work with Linux systems. 
1) Installation notes:
  - Install CUDA and GPU Acceleration (version 10 preferable) for NVIDIA cards, if applicable (CUDA 9 has been verified to work with updated Tensorflow v 1.5): https://developer.nvidia.com/cuda-10.0-download-archive. Note that the default install for NeMO-Net uses tensorflow-gpu. This can be changed in setup.py.
  - Install osgeo/gdal for geospatial manipulation (required for .shp, .gml, .gtif files). Refer to https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html for instructions. Also install_gdal.txt will have the commands to install the relevant gdal libraries.
  - Setup virtual environment as necessary, and run "pip install -r requirements.txt". Afterward, run "python setup.py install"

2) Images can be downloaded per request, and are to be contained within the ./Images/ folder. If choosing a different folder, please specify locations within the .yml files

3) Additional resources:
Code
  - VGG16 FCN code and base level code based upon: https://github.com/JihongJu/keras-fcn
  - ResNet architecture loosely based upon: https://github.com/raghakot/keras-resnet
  - Hyperas/ Hyperopt code: https://github.com/maxpumperla/hyperas
  - DeepLab code based upon: https://github.com/DrSleep/tensorflow-deeplab-lfov
  - pydensecrf code found here: https://github.com/lucasb-eyer/pydensecrf

 Papers
  - DeepLab: https://arxiv.org/abs/1606.00915
  - VGG FCN-like: http://www.mdpi.com/2072-4292/9/5/498
  - DCNNs with CRFs: https://arxiv.org/abs/1412.7062
  - VGG: https://arxiv.org/abs/1409.1556
  - ResNet: https://arxiv.org/abs/1512.03385
  - FCN: https://arxiv.org/abs/1411.4038
