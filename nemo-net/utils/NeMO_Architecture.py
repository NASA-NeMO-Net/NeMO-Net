from typing import Tuple, Callable, List, Union, Dict, Any

class NeMO_Architecture:
	def __init__(self, 
		conv_blocks: int = 0, 
		dense_blocks: int = 0):
		''' Initializes NeMO_Architecture with only encoder side params
			conv_blocks: # of major convolutional blocks. Each output of convolutional blocks may feed into a decoder block
			dense_blocks: # of dense blocks after all conv_blocks. Usually not applied for FCN
		'''

		self.conv_blocks = conv_blocks
		self.dense_blocks = dense_blocks
		self.conv_params = None

	def resnet50_defaultparams(self):
		''' Defines default resnet50 parameters. 5 major convolution blocks, 1st one being large filter and pooling layer, while the rest are parallel convolutional layers.
		This version does not have dense layers afterwards
		'''
		self.conv_blocks = 5
		self.dense_blocks = 0
		self.conv_params = {"filters": [[64] , [([64,64,128],128)]*3, [([128,128,256],256)]*4, [([256,256,512],512)]*6, [([512,512,1024],1024)]*3],
			"conv_size": [[(7,7)] , [([(1,1),(3,3),(1,1)], (1,1))]*3, [([(1,1),(3,3),(1,1)], (1,1))]*4, [([(1,1),(3,3),(1,1)], (1,1))]*6, [([(1,1),(3,3),(1,1)], (1,1))]*3],
			"conv_strides": [(2,2), [([(1,1),(1,1),(1,1)], (1,1))] + [(1,1)]*2 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*3 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*5, [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*2],
			"padding": ['same', 'same', 'same', 'same', 'same'],
			"dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1)],
			"pool_size": [(3,3), (1,1), (1,1), (1,1), (1,1)],
			"pool_strides": [(2,2), (1,1), (1,1), (1,1), (1,1)],
			"pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0)],
			"dropout": [None] * self.conv_blocks,
			"filters_up": [None] * self.conv_blocks,
			"upconv_size": [None] * self.conv_blocks,
			"upconv_strides": [None] * self.conv_blocks,
			"layercombo": ["cbap", [("cbacbac","c")]+[("bacbacbac","")]*2, [("bacbacbac","c")]+[("bacbacbac","")]*3, [("bacbacbac","c")]+[("bacbacbac","")]*5, [("bacbacbac","c")]+[("bacbacbac","")]*2], 
			"layercombine": ["","sum","sum","sum","sum"],           
			"dense_filters": [None], 
			"dense_dropout": [None]}

class NeMO_FCN_Architecture(NeMO_Architecture):
	
	def __init__(self,
		conv_blocks: int = 0,
		dense_blocks: int = 0,
		decoder_index: List[int] = None,
		scales: List[float] = None):
		''' Initializes NeMO_Architecture with encoder/decoder params
			conv_blocks: # of major convolutional blocks. Each output of convolutional blocks may feed into a decoder block
			dense_blocks: # of dense blocks after all conv_blocks. Usually not applied for FCN
			decoder_index: List of encoder output indices to take as decoder input (note that index 0 starts at the deepest layer i.e. final output of encoder)
			scales: List of scaling factors for every encoder output/ decoder input. Must be same length as decoder_index
		'''

		super().__init__(conv_blocks = conv_blocks, dense_blocks = dense_blocks)
		assert len(scales) == len(decoder_index), "Decoder index not same length as scales!"

		self.decoder_index = decoder_index
		self.scales = scales
		self.bridge_params = None
		self.prev_params = None
		self.next_params = None

	def refinemask_defaultparams(self):
		self.resnet50_defaultparams()
		self.decoder_index = [0,1,2,3]
		self.scales = [1,1,1,1]

		RCU = ("bacbac","")
		CRPx2 = ([("pbc",""),"pbc"],"")

		self.bridge_params = {"filters": [[1024,1024,128], [512,512,64], [256,256,32], [128,128,16]],
			"conv_size": [(3,3), (3,3), (3,3), (3,3)],
			"filters_up": [None]*4,
			"upconv_size": [None]*4,
			"upconv_strides": [None]*4,
			"layercombo": [[RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"]],
			"layercombine": ["sum","sum","sum","sum"]}

		self.prev_params = {"filters": [None, [128,128,64], [64,64,32], [32,32,16]],
			"conv_size": [None, (3,3),(3,3),(3,3)],
			"filters_up": [None,None,None,None],
			"upconv_size": [None,None,None,None],
			"upconv_strides": [None,(2,2),(2,2),(2,2)],
			"upconv_type": [None,"bilinear","bilinear","bilinear"],
			"layercombo": ["", [RCU,RCU,"bcu"], [RCU,RCU,"bcu"], [RCU,RCU,"bcu"]],
			"layercombine": [None,"sum","sum","sum"]} 

		self.next_params = {"filters": [["",128,128], ["",64,64], ["",32,32], ["",16,16,16,None,16,None]],
			"conv_size": [(3,3), (3,3), (3,3), (3,3)],
			"pool_size": [(5,5), (5,5), (5,5), (5,5)],
			"filters_up": [None,None,None,None],
			"upconv_size": [None,None,None,None],
			"upconv_strides": [None,None,None,(2,2)],
			"upconv_type": [None,None,None,"bilinear"],
			"layercombo": [["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU,RCU,"u",RCU,"u"]],
			"layercombine": ["sum","sum","sum","sum"]} 





