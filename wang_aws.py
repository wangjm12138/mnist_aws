#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import datetime
from my_minist import MNIST_W
import os
tf.logging.set_verbosity(tf.logging.INFO)
#import pdb
wjm_output="/opt/ml/model"
wjm_steps=20000

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# MNIST images are 28x28 pixels, and have one color channel
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
	
	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28, 28, 1]
	# Output Tensor Shape: [batch_size, 28, 28, 32]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 28, 28, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	
	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 14, 14, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 64]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 14, 14, 64]
	# Output Tensor Shape: [batch_size, 7, 7, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	
	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 7, 7, 64]
	# Output Tensor Shape: [batch_size, 7 * 7 * 64]
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	
	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 7 * 7 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	
	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 10]
	logits = tf.layers.dense(inputs=dropout, units=10)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	# Load training and eval data
	mnist_instance = MNIST_W(train_images_path='/home/train-images-idx3-ubyte',train_labels_path='/home/train-labels-idx1-ubyte', \
				test_images_path='/home/t10k-images-idx3-ubyte',test_labels_path='/home/t10k-labels-idx1-ubyte')
	mnist_instance.get_image()
	train_data = mnist_instance.train_images.astype(np.float32)
	train_data.resize(60000, 784)
	train_labels = mnist_instance.train_labels.astype(np.int32)
	eval_data = mnist_instance.test_images.astype(np.float32)
	eval_data.resize(10000, 784)
	eval_labels = mnist_instance.test_labels.astype(np.int32)
#	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#	train_data = mnist.train.images  # Returns np.array
#	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#	eval_data = mnist.test.images  # Returns np.array
#	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	print(len(train_data),type(train_data),train_data.dtype,train_data.shape,train_data.ndim)
	print(len(train_labels),type(train_labels),train_labels.dtype,train_labels.shape,train_labels.ndim)
	print(len(eval_data),type(eval_data),eval_data.dtype,eval_data.shape,eval_data.ndim)
	print(len(eval_labels),type(eval_labels),eval_labels.dtype,eval_labels.shape,eval_labels.ndim)
	#pdb.set_trace()
	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir=wjm_output)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
	tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
	shuffle=True)
	mnist_classifier.train(
		input_fn=train_input_fn,
		steps=wjm_steps,
	hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
		x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	tf.logging.info("end")
	tf.logging.info(time.strftime("%Y.%m.%d:%H.%M.%S",time.localtime(time.time())))
	endtime = datetime.datetime.now()
	last_time=(endtime-starttime).seconds
	os.system("echo %s > /opt/ml/model/a.txt"%str(last_time))
	print(eval_results)

if __name__ == "__main__":
	starttime = datetime.datetime.now()
	tf.logging.info("start")
	tf.logging.info(time.strftime("%Y.%m.%d:%H.%M.%S",time.localtime(time.time())))
	tf.app.run()
