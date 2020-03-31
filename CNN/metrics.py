import numpy as np
from sklearn.metrics import confusion_matrix


class metrics(object):
	'''
	This class takes the prediction of a model and the corresponding target data as a [samples, width, height, classes] 4-D vector
	and evaluates the results based on requested metric. Classes should start from 0.
	The class automatically takes care of "one-hot encoded" or "argmax"ed results.
	'''
	def __init__(self, y_obs, y_pred, classes=1):
		self.y_obs = np.array(y_obs)
		self.y_pred = np.array(y_pred)
		assert self.y_obs.ndim == 4 and self.y_pred.ndim == 4, "Shape of the data is not consistent with the class requirements."
		self.shape = self.y_obs.shape
		self.classes = classes
		assert len(np.unique(self.y_obs)) <= self.classes and len(np.unique(self.y_pred)) <= self.classes, "Number of classes are not correctly set ({} instead of {}).".format(max(len(np.unique(self.y_pred)), len(np.unique(self.y_pred))), self.classes)

		if self.y_obs.shape[-1] == 1:
			self.y_obs_one_hot = np.zeros(list(self.shape[:-1]) + [self.classes,])
			for i in range(self.classes):
				mask = self.y_obs == i
				self.y_obs_one_hot[mask[..., 0], i] = 1
		else:
			self.y_obs_one_hot = np.copy(self.y_obs)
			self.y_obs = np.array([np.argmax(yy, axsi=-1) for yy in self.y_obs])[..., np.newaxis]

		if self.y_pred.shape[-1] == 1:
			self.y_pred_one_hot = np.zeros(list(self.shape[:-1]) + [self.classes,])
			for i in range(self.classes):
				mask = self.y_pred == i
				self.y_pred_one_hot[mask[..., 0], i] = 1
		else:
			self.y_pred_one_hot = np.copy(self.y_pred)
			self.y_pred = np.array([np.argmax(yy, axis=-1) for yy in self.y_pred])[..., np.newaxis]


	def label_accuracy(self):
		ta = []
		for clss in range(self.classes):
			class_acc = np.sum(self.y_pred[self.y_obs==clss]==clss) / np.float(np.sum(self.y_obs==clss))
			ta.append(class_acc)
		return np.mean(np.delete(ta,6)), ta


	def print_confusionMatrix(self):
		return confusion_matrix(self.y_obs.reshape(-1, 1), self.y_pred.reshape(-1, 1), labels=range(self.classes))


	def intersect_of_union(self):
		area_of_overlap = []
		area_of_union = []
		for clss in range(self.classes):
			area_of_overlap.append(np.sum((self.y_obs==clss) & (self.y_pred==clss)))
			area_of_union.append(np.sum((self.y_obs==clss) + (self.y_pred==clss)))
		return np.array(area_of_overlap, dtype=float) / area_of_union


	def probability_of_detection(self):
		'''
		The probability of Detection (POD) is the fraction of observed events
		 that is forecast correctly. [This index is also known as Recall]
		'''
		tp, fn = [], []
		for clss in range(self.classes):
			tp.append(np.sum((self.y_obs==clss) & (self.y_pred==clss)))
			fn.append(np.sum((self.y_obs==clss) & (self.y_pred!=clss)))
		return [tpp * 1. / (tpp + fnn) for tpp, fnn in zip(tp, fn)]

	def false_alarm_ratio(self):
		'''
		The False Alarm Ratio (FAR) is the fraction of "yes" forecasts that 
		were wrong, i.e., were false alarms.
		'''
		tp, fp = [], []
		for clss in range(self.classes):
			tp.append(np.sum((self.y_obs==clss) & (self.y_pred==clss)))
			fp.append(np.sum((self.y_obs!=clss) & (self.y_pred==clss)))
		return [fpp * 1. / (tpp + fpp) for tpp, fpp in zip(tp, fp)]

	def critical_success_index(self):
		'''
		The Critical Success Index (CSI) combines POD and FAR into one score
		for low frequency events.
		'''
		tp, fp, fn = [], [], []
		for clss in range(self.classes):
			tp.append(np.sum((self.y_obs==clss) & (self.y_pred==clss)))
			fn.append(np.sum((self.y_obs==clss) & (self.y_pred!=clss)))
			fp.append(np.sum((self.y_obs!=clss) & (self.y_pred==clss)))
		return [tpp * 1. / (tpp + fpp + fnn) for tpp, fpp, fnn in zip(tp, fp, fn)]

	def rmse(self):
		rmse = []
		for clss in range(self.classes):
			rmse.append(np.sqrt(((self.y_pred - self.y_obs) ** 2).mean()))
		return rmse

	def precision(self):
		'''
		The Precision is the fraction of "yes" forecasts that were right.
		'''
		tp, fp = [], []
		for clss in range(self.classes):
			tp.append(np.sum((self.y_obs==clss) & (self.y_pred==clss)))
			fp.append(np.sum((self.y_obs!=clss) & (self.y_pred==clss)))
		return [tpp * 1. / (tpp + fpp) for tpp, fpp in zip(tp, fp)]
		
	def f1_score(self):
		score = [2. * (pr * pod) / (pr + pod) for pr, pod in zip(self.precision(), self.probability_of_detection())]
		return score


