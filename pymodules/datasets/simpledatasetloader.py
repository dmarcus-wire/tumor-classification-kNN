# import the necessary packages
# loop over all class labels and images
# return images and class labels to disk

import numpy as np # numerical array processing
import cv2 # opencv bindings
import os # for working with file paths on disk

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		# resizing or more advanced transformation
		# idea to chain preprocessors together
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	# laod dataset from disk, print verbose updates to terminal
	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		# data = raw pixel intensities
		data = []
		# labels = parse from the file path
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))