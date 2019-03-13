import tensorflow as tf 
import numpy as np
from PIL import Image
import sys
import os
import math

def show_progress(current, total, ps = ""):
	sys.stdout.write("\rprocessing %d / %d (%s)" % (current + 1, total, ps))
	sys.stdout.flush()

def int64_feature(value):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def bytes_feature(value):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def create_example(image, label):
	return tf.train.Example(features = tf.train.Features(feature = {
		"image": bytes_feature(image.tobytes()),
		"label": int64_feature(label)
		}))

def load_images_and_labels(input_dirs, image_type):
	img_paths = []
	labels = []

	for input_dir in input_dirs:
		dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
		for d in dirs:
			label = int(d)
			label_dir = os.path.join(input_dir, d)
			files = [f for f in os.listdir(label_dir) if f.endswith(".%s" % image_type)]
			for f in files:
				img_paths.append(os.path.join(label_dir, f))
				labels.append(label)
	return img_paths, labels

def encode(input_dirs, 
		   output_dir, 
		   image_length, 
		   grayscale = False, 
		   image_type = 'png', 
		   output_name = 'train', 
		   image_per_shard = 10000, 
		   shuffle = True):
	"""
		encode all the images in input_dirs to TFRecord file format
		it's supposed that images are in the dir which name is their label

	"""
	img_paths, labels = load_images_and_labels(input_dirs, image_type)
	img_count = len(img_paths)

	img_paths = np.array(img_paths)
	labels = np.array(labels)

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	if shuffle:
		indices = np.arange(img_count)
		np.random.shuffle(indices)
		img_paths = img_paths[indices]
		labels = labels[indices]

	write_times = math.ceil(img_count / image_per_shard)

	for i in range(write_times):
		record_path = os.path.join(output_dir, "%s_%d.record" % (output_name, i))
		if os.path.exists(record_path):
			os.remove(record_path)
		with tf.python_io.TFRecordWriter(record_path) as writer:
			for j in range(image_per_shard * i, min(image_per_shard * (i + 1), img_count)):
				total = min(image_per_shard * (i + 1), img_count) - image_per_shard * i
				current = j - image_per_shard * i
				show_progress(current, total, 'shard_%d' % i)
				fp = open(img_paths[j], 'rb')
				img = Image.open(fp).resize((image_length, image_length))
				if grayscale:
					img = img.convert('L')
				fp.close()

				example = create_example(img, labels[j])
				writer.write(example.SerializeToString())
			print(' done.')









