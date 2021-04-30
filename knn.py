# USAGE
# python knn.py --dataset dataset/tumors

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier # implements kNN algorithm
from sklearn.preprocessing import LabelEncoder # encodes strings as integers
from sklearn.model_selection import train_test_split # splittind data
from sklearn.metrics import classification_report # formatted report on accuracy
from pymodules.preprocessing import SimplePreprocessor # resize to fixed
from pymodules.datasets import SimpleDatasetLoader # loop over images in directory
from imutils import paths # easily extract names of images from directory
import argparse # cli arguments

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
# number of datapoints in surrounding area, you set this and change
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
# parallel jobs to run, using all available cores, scikit learn handles
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
# two tuple of data and labels
(data, labels) = sdl.load(imagePaths, verbose=500)
# reshape / flatten the data list into a 1D list of 3072 values = 32 * 32 * 3
# rows = total number of images columns = 3072
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
# larger the dataset, the more memory it requires
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
# pass in k and j from input
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
# trains kNN classifier, not really train, but copies data from x,y to classify
model.fit(trainX, trainY)
# testY = ground truth clsas labels
# model.predict to pass in test and predict
# against known classes
print(classification_report(testY, model.predict(testX),
	target_names=le.classes_))