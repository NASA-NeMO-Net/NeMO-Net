from setuptools import setup

setup(
	name = 'NASA NeMO-Net', 
	version = '0.1',
	description = 'Training and classification of Satellite Coral Imagery',
	author = 'Alan Li',
	packages=["nemo-net"],
	install_requires=[
		'numpy',
		'opencv-python',
		'matplotlib',
		'jupyter',
		'scikit-learn',
		'pandas',
		'Pillow',
		'GDAL',
		'pydensecrf',
		'networkx==1.11',
		'tensorflow-gpu==1.5.0'
		'keras==2.0.8',
	],
)