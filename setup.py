from setuptools import setup

setup(
	name = 'NASA NeMO-Net', 
	version = '0.1',
	description = 'Training and classification of Satellite Coral Imagery',
	author = 'Alan Li',
	packages=["nemo-net"],
	setup_requires=[
		'numpy==1.15.0',
		'scipy==1.1.0',
		'networkx==1.11',
		'cython>=0.28.5'
	],
	install_requires=[
		'matplotlib',
		'jupyter',
		'scikit-learn==0.19.2',
		'pyyaml',
		'Pillow',
		'pydensecrf',
		'tensorflow-gpu==1.13.1',
		'keras==2.0.8',
	],
)
