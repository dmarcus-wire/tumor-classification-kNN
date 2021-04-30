# import the necessary packages
# resize fixed width x fixed height
import cv2 #opencv bindings

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		# store the variables
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect ratio
		# because for kNN to work, each vector must have exact same dimensionality
		# in order to compute euclidean distance
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)