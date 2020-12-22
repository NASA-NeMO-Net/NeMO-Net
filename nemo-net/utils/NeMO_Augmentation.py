from typing import Tuple, Callable, List, Union, Dict
import abc
import numpy as np

from NeMO_Utils import apply_channel_corrections, normalize

class NeMOAugmentationModule(object):
	""" Augmentation basis object

	# Arguments
		num_channels: # of channels (e.g. 3 for RGB, 4 for Sentinel, 8 for WV-2)
		pixel_mean: Mean value to subtract during normalization per channel. Note that train images may already be normalized with (mean,std) = (100,100).
			This would add further normalization per channel. None implies a mean of 0
		pixel_std: Std value to normalize per channel. Note that train images may already be normalized with (mean,std) = (100,100).
			This would add further normalization per channel. None implies a std of 1
		reverse_normalize: whether to apply reverse normalization before augmentation. This is true for some modes (such as polynomial fit)
	"""
	def __init__(self,
		num_channels: int = 8,
		pixel_mean: Union[float, List[float]] = None,
		pixel_std: Union[float, List[float]] = None,
		reverse_normalize: bool = False):

		self.num_channels = num_channels
		self.pixel_mean = apply_channel_corrections(value = pixel_mean, 
			num_channels = self.num_channels, 
			id_str = "Augmentation pixel_mean")
		self.pixel_std = apply_channel_corrections(value = pixel_std, 
			num_channels = self.num_channels, 
			id_str = "Augmentation pixel_std")
		self.reverse_normalize = reverse_normalize

	@abc.abstractmethod
	def apply(self, input_image: np.ndarray) -> np.ndarray:
		"""
		Apply a certain augmentation transformation
		"""
		pass

	@staticmethod
	def random_flip_rotation(input_image: np.ndarray, 
		rnd_flip: bool = True,
		rnd_rotation: bool = True) -> Tuple[np.ndarray, int, int]:
		"""
		Applies random flip and rotation
			input_image: Image array (3D Tensor)
			rnd_flip: randomly flip if boolean
			rnd_rotation: randomly rotate if boolean
		"""
		x = input_image
		flip = 0
		num_rotate = 0

		if rnd_flip:
			flip = np.random.randint(0, 2) # randomize whether to flip or not
		if rnd_rotation:
			num_rotate = np.random.randint(0, 4) # randomize # of times to rotate by 90 degrees

		if flip:
			x = np.flip(x, axis = 0)
		x = np.rot90(x, k = num_rotate)

		return x, flip, num_rotate

	@staticmethod
	def flip_rotation(input_image: np.ndarray,
		flip: int = 0,
		rotation: int = 0) -> np.ndarray:
		"""
		Applies flip and rotation
			input_image: Image array (3D Tensor)
			flip: flip or not (integer value, 0 or 1)
			rotation: amount to rotate image by
		"""
		x = input_image
		if flip:
			x = np.flip(x, axis=0)
		x = np.rot90(x, k = rotation)

		return x

class PolynomialAugmentation(NeMOAugmentationModule):

	def __init__(self,
			num_channels: int = 4,
			pixel_mean: Union[float, List[float]] = 100.0,
			pixel_std: Union[float, List[float]] = 100.0,
			reverse_normalize: bool = True):

		super(PolynomialAugmentation, self).__init__(num_channels, pixel_mean, pixel_std, reverse_normalize)

		'''
		These are hard coded values for a polynomial fit based upon some tests of WV-2 data, stored in dictionary
		Assumes input channels are BGR + NIR (in that order)
		Polynomial fit is applied as:
			B_new = (f00*B^2 + f01*B + f02*G^2 + f03*G + f04*R^2 + f05*R + f06*NIR^2 + f07*NIR + f08 + random0)
			G_new = (f10*B^2 + f11*B + f12*G^2 + f13*G + f14*R^2 + f15*R + f16*NIR^2 + f17*NIR + f18 + random1) 
			R_new = (f20*B^2 + f21*B + f22*G^2 + f03*G + f24*R^2 + f25*R + f26*NIR^2 + f27*NIR + f28 + random2) 
			NIR_new = (f30*B^2 + f31*B + f32*G^2 + f33*G + f34*R^2 + f35*R + f36*NIR^2 + f37*NIR + f38 + random3) 
		'''
		self.fit = {}
		self.fit[1] = np.asarray([[-1.14243211e-02,1.74006653e+00,-7.41971633e-04,-2.27847582e-01,7.81802240e-03,-2.37788745e-01,0,0,2.64154330e+01],
			[9.11148733e-04,-1.46364262e+00,-4.90822080e-03,1.46584766e-01,1.48671030e-03,1.63453697e+00,0,0,8.56772131e+01],
			[7.47928597e-03,-2.45992530e+00,-3.75490029e-03,3.24331424e-01,-2.39503953e-03,1.55765285e+00,0,0,1.05204936e+02],
			[0,0,0,0,0,0,-0.05250297,2.22991288,4.85164778]])
		self.fit[2] = np.asarray([[-2.04295343e-02,3.64489475e+00,8.26687273e-03,-8.41662523e-01,9.44963864e-03,-6.06050531e-01,0,0,-5.16613993e+01],
			[-1.34827256e-02,1.68043246e+00,6.72071243e-03,-7.05877090e-01,5.27804649e-03,7.43471236e-01,0,0,-3.13653270e+01],
			[-6.07675802e-03,6.17908072e-01,6.08938030e-03,-4.44539262e-01,7.92493705e-04,6.39467461e-01,0,0,-1.17938458e+01],
			[0,0,0,0,0,0,-2.80239868e-03,1.69880579e-01,8.63254268e+00]])
		self.fit[3] = np.asarray([[-1.16126391e-02,2.23735380e+00,-1.40344430e-02,3.06320353e-02,1.32410016e-02,-6.57365715e-01,0,0,-2.30938626e+01],
			[-6.15389359e-03,1.16689906e+00,-1.14967466e-02,2.74771312e-02,5.29923732e-03,6.66334701e-01,0,0,-4.25799090e+01],
			[4.53015635e-03,-9.03508160e-01,5.52248862e-04,4.42759776e-02,-8.35786318e-03,1.68134937e+00,0,0,-1.15354561e+00],
			[0,0,0,0,0,0,-0.02869423,1.04414848,-2.43072596]])

	def apply(self, input_image: np.ndarray) -> np.ndarray:

		# Reverse-normalize
		x = normalize(input_array = input_image,
			pixel_mean = self.pixel_mean,
			pixel_std = self.pixel_std,
			reverse_normalize = self.reverse_normalize)
		
		x = np.rollaxis(x, 2, 0) # make channel_axis the first axis (makes fit easier to work with)

		white_pixels = np.where(np.sqrt(x[0]**2+x[1]**2+x[2]**2) >= 250) # find white pixels
		NIR_pixels = np.where(x[3] >= 50) # find NIR pixels (for land)
		toremap = np.random.randint(0, len(self.fit)+1) # randomize the polynomial fit (0 => no fit)

		xorig = np.copy(x)
		if toremap > 0: # if we are going to polynomial fit
			for i in range(self.num_channels):
				x[i] = self.apply_polynomial_fit2channel(xorig, self.fit[toremap], i)
				x[i][white_pixels] = xorig[i][white_pixels] # reset white pixels
				x[i][NIR_pixels] = xorig[i][NIR_pixels] # reset NIR pixels

		# Normalize
		x = np.rollaxis(x, 0, 3) # make channel_axis last axis again
		x = normalize(input_array = x,
			pixel_mean = self.pixel_mean,
			pixel_std = self.pixel_std,
			reverse_normalize = (not self.reverse_normalize)) 

		return x

	@staticmethod
	def apply_polynomial_fit2channel(x: np.ndarray, p: np.ndarray, idx: int) -> np.ndarray:
		# x: input image 
		# p: numpy array to remap by
		# idx: BGR + NIR index to remap
		channel = p[idx,0]*x[0]**2 + p[idx,1]*x[0] + \
			p[idx,2]*x[1]**2 + p[idx,3]*x[1] + \
			p[idx,4]*x[2]**2 + p[idx,5]*x[2] + \
			p[idx,6]*x[3]**2 + p[idx,7]*x[3] + \
			p[idx,8] + np.random.uniform(-0.1,0.1,(x.shape[1],x.shape[2]))

		return channel




