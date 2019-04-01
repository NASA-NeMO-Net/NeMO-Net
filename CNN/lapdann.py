import numpy as np
import tensorflow as tf
from flip_gradient import flip_gradient

np.random.seed(999)
tf.set_random_seed(999)


class LAPDANN(object):
	'''
	The following LAPDANN is a combined model which generates multimodal high-res classifications from various satellite data.
	'''
	def __init__(self, width=80, height=80, channels=4, classes=10, batch_size=1, scale_list=[80, 200, 400]):
		self.width = width
		self.height = height
		self.channels = channels
		self.shape = [width, height, channels]
		self.classes = classes
		self.batch_size = batch_size
		self.learning_rate = tf.placeholder(tf.float32, [])

		# To make sure that scales are correctly set.
		self.scale_list = sorted(scale_list)

		# Optional
		self.w_init = None
		self.w_reg = None

		# Variables for DANN
		self.domain = tf.placeholder(tf.float32, [None, 2])
		self.l = tf.placeholder(tf.float32, [])
		self.train_st = tf.placeholder(tf.bool, [])
		self.train_bn = tf.placeholder(tf.bool, [])

		# Model configs.
		self.lapgan_build()
		vars = [var for var in self.train_vars if var.name.startswith("dann_")]
		self.saver_dann = tf.train.Saver(vars)
		vars = [var for var in self.train_vars if (var.name.startswith("lapgan_generator_".format(scale_list[1])) or var.name.endswith("lapgan_discriminator_{}".format(scale_list[1])))]
		self.saver_lapgan_0 = tf.train.Saver(vars)
		vars = [var for var in self.train_vars if (var.name.startswith("lapgan_generator_{}".format(scale_list[2])) or var.name.endswith("lapgan_discriminator_{}".format(scale_list[2])))]
		self.saver_lapgan_1 = tf.train.Saver(vars)

	def dann_build(self, inp, out=None, reuse=None):
		# self.X = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels])
		# self.y = tf.placeholder(tf.float32, [None, self.width, self.height, self.classes])

		feature_conv0 = 64
		feature_conv1 = 96
		feature_conv2 = 128
		feature_conv3 = 160
		label_predictor = 100
		domain_predictor = 100

		# Feature Extractor
		with tf.variable_scope('dann_fe', reuse=reuse):
			fe_conv0 = tf.layers.conv2d(inp, filters=feature_conv0, kernel_size=5, strides=1, padding='same')
			fe_bn0 = tf.layers.batch_normalization(fe_conv0, training=self.train_bn)
			fe_act0 = tf.nn.relu(fe_bn0)

			fe_conv1 = tf.layers.conv2d(fe_act0, filters=feature_conv1, kernel_size=5, strides=2, padding='same')
			fe_bn1 = tf.layers.batch_normalization(fe_conv1, training=self.train_bn)
			fe_act1 = tf.nn.relu(fe_bn1)

			fe_conv2 = tf.layers.conv2d(fe_act1, filters=feature_conv2, kernel_size=5, strides=2, padding='same')
			fe_bn2 = tf.layers.batch_normalization(fe_conv2, training=self.train_bn)
			fe_act2 = tf.nn.relu(fe_bn2)

			fe_conv3 = tf.layers.conv2d(fe_act2, filters=feature_conv3, kernel_size=5, strides=2, padding='same')
			fe_bn3 = tf.layers.batch_normalization(fe_conv3, training=self.train_bn)
			fe_act3 = tf.nn.relu(fe_bn3)

			self.feature = tf.reshape(fe_act3, [-1, self.width/8 * self.height/8 * feature_conv3])

		# Segment Estimator
		with tf.variable_scope('dann_se', reuse=reuse):
            
			# Switches to route target examples (second half of batch) differently
			# depending on train or test mode.
			all_features = lambda: self.feature
			source_features = lambda: tf.slice(self.feature, [0, 0], [self.batch_size/2, -1])
			segment_feats = tf.cond(self.train_st, source_features, all_features)

			all_labels = lambda: out
			source_labels = lambda: tf.slice(out, [0, 0, 0, 0], [self.batch_size/2, -1, -1, -1])
			self.segment_labels = tf.cond(self.train_st, source_labels, all_labels)
			self.segment_labels = tf.reshape(self.segment_labels, [-1, self.classes])

			segment_feats = tf.reshape(segment_feats, [-1, self.width/8, self.height/8, feature_conv3])

			se_conv0 = tf.layers.conv2d_transpose(segment_feats, filters=feature_conv3, kernel_size=5, strides=2, padding='same')
			se_bn0 = tf.layers.batch_normalization(se_conv0, training=self.train_bn)
			se_act0 = tf.nn.relu(se_bn0)

			tmp = tf.cond(self.train_st, lambda: tf.slice(fe_act2, [0, 0, 0, 0], [self.batch_size/2, -1, -1, -1]), lambda: fe_act2)
			se_act0 = tf.concat([tmp, se_act0], -1)
			se_conv1 = tf.layers.conv2d_transpose(se_act0, filters=feature_conv2, kernel_size=5, strides=2, padding='same')
			se_bn1 = tf.layers.batch_normalization(se_conv1, training=self.train_bn)
			se_act1 = tf.nn.relu(se_bn1)

			tmp = tf.cond(self.train_st, lambda: tf.slice(fe_act1, [0, 0, 0, 0], [self.batch_size/2, -1, -1, -1]), lambda: fe_act1)
			se_act1 = tf.concat([tmp, se_act1], -1)
			se_conv2 = tf.layers.conv2d_transpose(se_act1, filters=feature_conv1, kernel_size=5, strides=2, padding='same')
			se_bn2 = tf.layers.batch_normalization(se_conv2, training=self.train_bn)
			se_act2 = tf.nn.relu(se_bn2)

			tmp = tf.cond(self.train_st, lambda: tf.slice(fe_act0, [0, 0, 0, 0], [self.batch_size/2, -1, -1, -1]), lambda: fe_act0)
			se_act2 = tf.concat([tmp, se_act2], -1)
			se_conv3 = tf.layers.conv2d_transpose(se_act2, filters=feature_conv0, kernel_size=5, strides=1, padding='same')
			se_bn3 = tf.layers.batch_normalization(se_conv3, training=self.train_bn)
			se_act3 = tf.nn.relu(se_bn3)

			se_conv4 = tf.layers.conv2d_transpose(se_act3, filters=self.classes, kernel_size=5, strides=1, padding='same')

			# This step is to get softmax work as desired.
			se_conv4 = tf.reshape(se_conv4, [-1, self.classes])

#			pred = tf.reshape(tf.nn.softmax(se_conv4), [-1, self.width, self.height, self.classes])
#			self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=se_conv4, labels=self.segment_labels)
#			segment_labels = tf.reshape(segment_labels, [-1, self.width, self.height, self.classes])

		# Domain Predictor
		with tf.variable_scope('dann_dp', reuse=reuse):
            
			# Flip the gradient when backpropagating through this operation
			feat = flip_gradient(self.feature, self.l)
            
			dp_fc0 = tf.layers.dense(feat, units=domain_predictor)
			dp_bn0 = tf.layers.batch_normalization(dp_fc0, training=self.train_bn)
			dp_ac0 = tf.nn.relu(dp_bn0)

			dp_fc1 = tf.layers.dense(dp_ac0, units=domain_predictor)
			dp_bn1 = tf.layers.batch_normalization(dp_fc1, training=self.train_bn)
			dp_ac1 = tf.nn.relu(dp_bn1)

			dp_fc2 = tf.layers.dense(dp_ac1, units=domain_predictor)
			dp_bn2 = tf.layers.batch_normalization(dp_fc2, training=self.train_bn)
			dp_ac2 = tf.nn.relu(dp_bn2)


			d_logits = tf.layers.dense(dp_ac2, units=2)

#			domain_pred = tf.nn.softmax(d_logits)
#			self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)
		return se_conv4, d_logits


	def generator2_build(self, inp, condition=None, scale=200, reuse=None):
		with tf.variable_scope("lapgan_generator_{}".format(scale), reuse=reuse):
			if condition is not None:
				inp = tf.concat([inp, condition], axis=3)
			x = tf.layers.conv2d(inp, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="g_conv_01")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="g_conv_02")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.conv2d(x, filters=self.classes, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="g_conv_03")
			return x


	def generator1_build(self, inp, condition=None, scale=400, reuse=None):
		with tf.variable_scope("lapgan_generator_{}".format(scale), reuse=reuse):
			if condition is not None:
				inp = tf.concat([inp, condition], axis=3)
			x = tf.layers.conv2d(inp, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="g_conv_01")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="g_conv_02")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.conv2d(x, filters=self.classes, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="g_conv_03")
			return x


	def discriminator2_build(self, inp, diff=None, condition=None, scale=200, reuse=None):
		with tf.variable_scope("lapgan_discriminator_{}".format(scale), reuse=reuse):
			inp += diff
			if condition is not None:
				inp = tf.concat([inp, condition], axis=3)
			x = tf.layers.conv2d(inp, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="d_conv_01")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.dropout(x, 0.5, name="d_dropout_01")

			x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="d_conv_02")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.dropout(x, 0.5, name="d_dropout_02")

			x = tf.reshape(x, [-1, scale * scale * 64])
			x = tf.layers.dense(x, 1, kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="d_dense_04")
			return x

	def discriminator1_build(self, inp, diff=None, condition=None, scale=400, reuse=None):
		with tf.variable_scope("lapgan_discriminator_{}".format(scale), reuse=reuse):
			inp += diff
			if condition is not None:
				inp = tf.concat([inp, condition], axis=3)
			x = tf.layers.conv2d(inp, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="d_conv_01")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.dropout(x, 0.5, name="d_dropout_01")

			x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="d_conv_02")
			# x = tf.layers.batch_normalization(x, training=self.train_bn)
			x = tf.nn.relu(x)
			x = tf.layers.dropout(x, 0.5, name="d_dropout_02")

			x = tf.reshape(x, [-1, scale * scale * 64])
			x = tf.layers.dense(x, 1, kernel_initializer=self.w_init, kernel_regularizer=self.w_reg, name="d_dense_04")
			return x

	def lapgan_build(self):
		self.img_X0 = tf.placeholder(tf.float32, shape=[None, ] + self.shape, name="Low_Res_Input") # Input [80 x 80IL]
		shape = [self.scale_list[2], self.scale_list[2], self.channels]
		self.img_Z0 = tf.placeholder(tf.float32, shape=[None, ] + shape, name="Full_Res_Input") # Input [400 x 400IH]
		shape = [self.scale_list[2], self.scale_list[2], self.classes]
		self.img_Y0 = tf.placeholder(tf.float32, shape=[None, ] + shape, name="Full_Res_Output") # Output [400 x 400OH]

		self.img_Z1 = Resize(self.img_Z0, size=[self.scale_list[0], self.scale_list[0]])	# resize down [80 x 80]

		self.img_H1 = Resize(self.img_Y0, size=[self.scale_list[1], self.scale_list[1]])	# resize down [200 x 200]
		self.img_H2 = Resize(self.img_H1, size=[self.scale_list[0], self.scale_list[0]])	# resize down [80 x 80]
		
		self.img_L1 = Resize(self.img_H1, size=[self.scale_list[2], self.scale_list[2]])	# resize up [400 x 400]
		self.img_L2 = Resize(self.img_H2, size=[self.scale_list[1], self.scale_list[1]])	# resize up [200 x 200]
		
		self.img_diff_1 = self.img_Y0 - self.img_L1						# [400 x 400]
		self.img_diff_2 = self.img_H1 - self.img_L2						# [200 x 200]


		# First pyramid [400x400]
		gen_1 = self.generator1_build(inp=self.img_L1, condition=None, scale=self.scale_list[-1])
		disc_1_fake = self.discriminator1_build(inp=gen_1, diff=self.img_L1, condition=None, scale=self.scale_list[-1])
		disc_1_real = self.discriminator1_build(inp=self.img_diff_1, diff=self.img_L1, condition=None, scale=self.scale_list[-1], reuse=True)
		# Second pyramid [200x200]
		gen_2 = self.generator2_build(inp=self.img_L2, condition=None, scale=self.scale_list[-2])
		disc_2_fake = self.discriminator2_build(inp=gen_2, diff=self.img_L2, condition=None, scale=self.scale_list[-2])
		disc_2_real = self.discriminator2_build(inp=self.img_diff_2, diff=self.img_L2, condition=None, scale=self.scale_list[-2], reuse=True)
		# Last pyramid [80x80]
		print(np.vstack([self.img_Z1, self.img_X0]).shape)
		self.gen_dann, disc_dann = self.dann_build(inp=tf.concat([self.img_X0, self.img_Z1], axis=0), out=tf.concat([self.img_H2, self.img_H2], axis=0))

		# Model parts
		self.gs = [gen_1, gen_2]
		self.d_reals = [disc_1_real, disc_2_real]
		self.d_fakes = [disc_1_fake, disc_2_fake]

		# Loss definition
		self.g_loss, self.d_loss = [], []
		with tf.variable_scope("loss"):
			for disc_f, disc_r in zip(self.d_fakes, self.d_reals):
				d_loss = bce_loss(disc_r, tf.ones_like(disc_r)) + bce_loss(disc_f, tf.zeros_like(disc_f))
				self.d_loss.append(d_loss)
				g_loss = bce_loss(disc_f, tf.ones_like(disc_f))
				self.g_loss.append(g_loss)

			# # DANN optimization
			# Define the loss functions.
			dann_c_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.gen_dann, labels=self.segment_labels)
			dann_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=disc_dann, labels=self.domain)
			dann_c_loss = tf.reduce_mean(dann_c_loss)
			dann_d_loss = tf.reduce_mean(dann_d_loss)
			self.dann_total_loss = dann_c_loss + dann_d_loss

		# Backprop step
		self.d_step = []
		self.g_step = []
		self.train_vars = tf.trainable_variables()
		for idx, scale in enumerate(self.scale_list[:0:-1]):
			# Define the training steps for Discriminators of LAPGAN.
			vars = [var for var in self.train_vars if var.name.startswith("lapgan_discriminator_{}".format(scale))]
			d_step = tf.train.AdamOptimizer(learning_rate=8e-4, beta1 = 0.5, beta2=0.9).minimize(self.d_loss[idx], var_list=vars)
			self.d_step.append(d_step)

			# Define the training steps for Generators of LAPGAN.
			vars = [var for var in self.train_vars if var.name.startswith("lapgan_generator_{}".format(scale))]
			g_step = tf.train.AdamOptimizer(learning_rate=8e-4, beta1 = 0.5, beta2=0.9).minimize(self.g_loss[idx], var_list=vars)	# self.learning_rate*
			self.g_step.append(g_step)

		# Define the training steps for DANN.
		vars = [var for var in self.train_vars if var.name.startswith("dann")]
		self.dann_train_op = tf.train.AdamOptimizer(learning_rate=8e-4, beta1 = 0.5, beta2=0.9).minimize(self.dann_total_loss, var_list=vars)
#		self.dann_train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.002).minimize(self.dann_total_loss, var_list=vars)
		self.BN_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			
			
	def train(self, sess, trainX, trainDX, trainY, epochs=1, batch_size=1, shuffle=False, verbose=True):
		
		domain_labels = np.vstack([np.tile([1., 0.], [self.batch_size/2, 1]), np.tile([0., 1.], [self.batch_size/2, 1])])
		num_batches = trainX.shape[0] / self.batch_size
		for epoch in range(epochs):
			data_batch = self.batch_generator([trainX, trainDX, trainY], batch_size=self.batch_size/2, shuffle=shuffle)
#			data_batch = self.batch_generator([trainX, trainY], batch_size=batch_size)
			for batch_num, (batchX, batchDX, batchY) in enumerate(data_batch):
				batch_no = batch_num + (epoch * num_batches)
				# Adaptation param and learning rate schedule as described in the paper
				p = float(batch_no) / (epochs * num_batches)
				l = 2. / (1. + np.exp(-10. * p)) - 1.
				lr = 0.01 / (1. + 10 * p)**0.75

				feed_dict = {self.img_Z0: batchX, self.img_X0: batchDX, self.img_Y0: batchY, self.domain: domain_labels, self.train_st: True, self.train_bn: True, self.l: l, self.learning_rate: lr}

				# Forward pass
#				img_L0, img_H1, img_L1, img_H2, img_L2, img_H3 = sess.run([self.img_L0, self.img_H1, self.img_L1, self.img_H2, self.img_L2, self.img_H3], feed_dict=feed_dict)
#				img_dann, img_gen2, img_gen3 = sess.run([self.gen_dann, .self.gs[0], self.gs[1]], feed_dict=feed_dict)
#				img_dann = tf.reshape(tf.nn.softmax(img_dann), [-1, self.width, self.height, self.classes])

				# Backprop
				_, _, _, _, _, _ = sess.run([self.dann_train_op, self.BN_update, self.d_step[0], self.g_step[0], self.d_step[1], self.g_step[1]], feed_dict=feed_dict)
				# Losses
				total_loss_dann, g_loss_1, g_loss_2, d_loss_1, d_loss_2 = sess.run([self.dann_total_loss, self.g_loss[0], self.g_loss[1], self.d_loss[0], self.d_loss[1]], feed_dict=feed_dict)

			if verbose:
				d_loss = (d_loss_1 + d_loss_2) / 2.
				g_loss = (g_loss_1 + g_loss_2) / 2.
				print("Epoch {}/{}".format(epoch+1, epochs))
				print("[DANN batch loss: {0:.3f}] [Dicriminator loss: {1:.3f}] - [Generator loss: {2:.3f}]".format(total_loss_dann, d_loss_1, d_loss_2, g_loss_1, g_loss_2))


	def train_dann_only(self, sess, trainX, trainDX, trainY, epochs=1, batch_size=1, shuffle=False, verbose=True):
		
		domain_labels = np.vstack([np.tile([1., 0.], [self.batch_size/2, 1]), np.tile([0., 1.], [self.batch_size/2, 1])])
		num_batches = trainX.shape[0] / self.batch_size
		for epoch in range(epochs):
			data_batch = self.batch_generator([trainX, trainDX, trainY], batch_size=self.batch_size/2, shuffle=shuffle)
			for batch_num, (batchX, batchDX, batchY) in enumerate(data_batch):
				batch_no = batch_num + (epoch * num_batches)
				# Adaptation param and learning rate schedule as described in the paper
				p = float(batch_no) / (epochs * num_batches)
				l = 2. / (1. + np.exp(-10. * p)) - 1.
				lr = 0.01 / (1. + 10 * p)**0.75

				feed_dict = {self.img_Z0: batchX, self.img_X0: batchDX, self.img_Y0: batchY, self.domain: domain_labels, self.train_st: True, self.train_bn: True, self.l: l, self.learning_rate: lr}
				# Backprop
				_, _ = sess.run([self.dann_train_op, self.BN_update], feed_dict=feed_dict)
				# Losses
				total_loss_dann = sess.run(self.dann_total_loss, feed_dict=feed_dict)

			if verbose:
				print("Epoch {}/{}".format(epoch+1, epochs))
				print("[DANN batch loss: {0:.3f}]".format(total_loss_dann))


	def train_lapgan_only(self, sess, trainY, epochs=1, batch_size=1, shuffle=False, verbose=True):
		
		domain_labels = np.vstack([np.tile([1., 0.], [self.batch_size/2, 1]), np.tile([0., 1.], [self.batch_size/2, 1])])
		num_batches = trainX.shape[0] / self.batch_size
		for epoch in range(epochs):
			data_batch = self.batch_generator([trainY, trainY], batch_size=self.batch_size/2, shuffle=shuffle)
			for batch_num, (batchY, _) in enumerate(data_batch):
				batch_no = batch_num + (epoch * num_batches)
				# Adaptation param and learning rate schedule as described in the paper
				p = float(batch_no) / (epochs * num_batches)
				l = 2. / (1. + np.exp(-10. * p)) - 1.
				lr = 0.01 / (1. + 10 * p)**0.75

				feed_dict = {self.img_Z0: batchX, self.img_X0: batchDX, self.img_Y0: batchY, self.domain: domain_labels, self.train_st: True, self.train_bn: True, self.l: l, self.learning_rate: lr}
				# Backprop
				_, _, _, _, _ = sess.run([self.BN_update, self.d_step[0], self.g_step[0], self.d_step[1], self.g_step[1]], feed_dict=feed_dict)
				# Losses
				g_loss_1, g_loss_2, d_loss_1, d_loss_2 = sess.run([self.g_loss[0], self.g_loss[1], self.d_loss[0], self.d_loss[1]], feed_dict=feed_dict)

			if verbose:
				d_loss = (d_loss_1 + d_loss_2) / 2.
				g_loss = (g_loss_1 + g_loss_2) / 2.
				print("Epoch {}/{}".format(epoch+1, epochs))
				print("[Dicriminator loss: {1:.3f}] - [Generator loss: {2:.3f}]".format(d_loss_1, d_loss_2, g_loss_1, g_loss_2))


	def batch_generator(self, data, batch_size=1, shuffle=False):
		def shuffle_helper(data):
			num = data[0].shape[0]
			p = list(np.random.permutation(num))
			return [d[p] for d in data]
		
		if shuffle:
			data = shuffle_helper(data)
		
		batch_count = 0
		while batch_count * batch_size + batch_size < len(data[0]):
			start = batch_count * batch_size
			end = start + batch_size
			batch_count += 1
			yield [d[int(start):int(end)] for d in data]

	def test(self, sess, testX, batch_size=1, verbose=False):

		# Define the models.
		img_H2 = tf.placeholder(tf.float32, shape=[None, self.scale_list[0], self.scale_list[0], self.classes])
		if testX.shape[1] == self.scale_list[2]:
			img_Z0 = tf.placeholder(tf.float32, shape=[None, self.scale_list[2], self.scale_list[2], self.channels])
			img_Z1 = Resize(img_Z0, [self.scale_list[0], self.scale_list[0]])
		elif testX.shape[1] == self.scale_list[0]:
			img_Z1 = tf.placeholder(tf.float32, shape=[None, self.scale_list[0], self.scale_list[0], self.channels])


		dann_sc0, _ = self.dann_build(inp=img_Z1, out=img_H2, reuse=True)
		dann_sc0 = tf.reshape(tf.nn.softmax(dann_sc0), [-1, self.width, self.height, self.classes])
		dann_sc1 = Resize(dann_sc0, [self.scale_list[1], self.scale_list[1]])

		gan_sc1 = self.generator2_build(inp=dann_sc1, condition=None, scale=self.scale_list[1], reuse=True)
		result_sc1 = gan_sc1 + dann_sc1
		dann_sc2 = Resize(result_sc1, [self.scale_list[2], self.scale_list[2]])

		gan_sc2 = self.generator1_build(inp=dann_sc2, condition=None, scale=self.scale_list[2], reuse=True)
		result_sc2 = gan_sc2 + dann_sc2

		data_batch = self.batch_generator([testX, testX], batch_size=self.batch_size, shuffle=False)
		predX = np.zeros([testX.shape[0], self.scale_list[2], self.scale_list[2], self.classes])
		predDANN = np.zeros([testX.shape[0], self.scale_list[0], self.scale_list[0], self.classes])
		dummy_out = np.zeros((self.batch_size, self.scale_list[0], self.scale_list[0], self.classes))
		for batch_num, (batchX, _) in enumerate(data_batch):
			if testX.shape[1] == self.scale_list[2]:
				feed_dict = {img_Z0: batchX, img_H2: dummy_out, self.train_st: False, self.train_bn: False}
			elif testX.shape[1] == self.scale_list[0]:
				feed_dict = {img_Z1: batchX, img_H2: dummy_out, self.train_st: False, self.train_bn: False}
			d_sc2, d_sc1, d_sc0 = sess.run([result_sc2, result_sc1, dann_sc0], feed_dict=feed_dict)
			predX[batch_num*self.batch_size:(batch_num+1)*self.batch_size] = d_sc2
			predDANN[batch_num*self.batch_size:(batch_num+1)*self.batch_size] = d_sc0
		return predX, predDANN


	def test_lapgan(self, sess, testY, batch_size=1, verbose=False):

		# Define the models.
		img_H3 = tf.placeholder(tf.float32, shape=[None, self.scale_list[2], self.scale_list[2], self.classes])
		img_H2 = Resize(img_H3, [self.scale_list[0], self.scale_list[0]])
		img_L1 = Resize(img_H2, [self.scale_list[1], self.scale_list[1]])

		gan_sc1 = self.generator2_build(inp=img_L1, condition=None, scale=self.scale_list[1], reuse=True)
		result_sc1 = gan_sc1 + img_L1
		dann_sc2 = Resize(result_sc1, [self.scale_list[2], self.scale_list[2]])

		gan_sc2 = self.generator1_build(inp=dann_sc2, condition=None, scale=self.scale_list[2], reuse=True)
		result_sc2 = gan_sc2 + dann_sc2

		data_batch = self.batch_generator([testY, testY], batch_size=self.batch_size, shuffle=False)
		predY0 = np.zeros([testY.shape[0], self.scale_list[2], self.scale_list[2], self.classes])
		predY1 = np.zeros([testY.shape[0], self.scale_list[1], self.scale_list[1], self.classes])
		predY2 = np.zeros([testY.shape[0], self.scale_list[0], self.scale_list[0], self.classes])
		for batch_num, (batchY, _) in enumerate(data_batch):
			feed_dict = {img_H3: batchY, self.train_st: False, self.train_bn: False}
			d_sc2, d_sc1, d_sc0 = sess.run([result_sc2, result_sc1, img_H2], feed_dict=feed_dict)
			predY0[batch_num*self.batch_size:(batch_num+1)*self.batch_size] = d_sc2
			predY1[batch_num*self.batch_size:(batch_num+1)*self.batch_size] = d_sc1
			predY2[batch_num*self.batch_size:(batch_num+1)*self.batch_size] = d_sc0
		return predY0, predY1, predY2
			
	def save_model(self, sess, filepath="./checkpoint"):
		self.saver_dann.save(sess, filepath+"_dann.ckpt")
		self.saver_lapgan_0.save(sess, filepath+"_lapgan_0.ckpt")
		self.saver_lapgan_1.save(sess, filepath+"_lapgan_1.ckpt")
		print("Model is successfully saved at {}.".format(filepath))

	def save_dann(self, sess, filepath="./checkpoint"):
		self.saver_dann.save(sess, filepath+"_dann.ckpt")
		print("DANN is successfully saved at {}.".format(filepath))

	def save_lapgan(self, sess, filepath="./checkpoint"):
		self.saver_lapgan_0.save(sess, filepath+"_lapgan_0.ckpt")
		self.saver_lapgan_1.save(sess, filepath+"_lapgan_1.ckpt")
		print("LAPGAN is successfully saved at {}.".format(filepath))
		
	def load_model(self, sess, filepath):
		self.saver_dann.restore(sess, filepath+"_dann.ckpt")
		self.saver_lapgan_0.restore(sess, filepath+"_lapgan_0.ckpt")
		self.saver_lapgan_1.restore(sess, filepath+"_lapgan_1.ckpt")
		print("Model is successfully restored.")

	def load_dann(self, sess, filepath):
		self.saver_dann.restore(sess, filepath+"_dann.ckpt")
		print("DANN is successfully restored.")

	def load_lapgan(self, sess, filepath):
		self.saver_lapgan_0.restore(sess, filepath+"_lapgan_0.ckpt")
		self.saver_lapgan_1.restore(sess, filepath+"_lapgan_1.ckpt")
		print("LAPGAN is successfully restored.")

def Resize(img, size=[1, 1], interpolation=tf.image.ResizeMethod.BILINEAR):
	return tf.image.resize_images(img, size, interpolation)
	
def bce_loss(logits, labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

