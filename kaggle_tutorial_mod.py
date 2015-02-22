# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:31:42 2015
This is adapted from the kaggle tutorial for the National Data Science Bowl at
https://www.kaggle.com/c/datasciencebowl/details/tutorial
Any code section lifted from the tutorial will start with # In tutorial [n].
My adaption will start with # Adapted

2/21/2015: I skipped over all the buildup stuff and just used the functions
they summarized it in. Works fine.
DONE: 1. Create a file with all that other stuff removed.
DONE: 2. Then adapt the file references back to what Chris simplified it to and
make sure it runs.  
3. Write/adapt the code to build a submission
4. Figure out how this fits with CNNs.

@author: kperez-lopez
"""
# In tutorial [1]:

from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max


# make graphics inline
# Editor says this is an error.
# TODO: figure out why.
# %matplotlib inline -I've got IPython console set to "automatic," which yields
# a separate window for graphics

# In tutorial [2]:  ( don't know why they include this)
import warnings
warnings.filterwarnings("ignore")


path = "./"

# Using .differences(...) removes any files in the dir train that have,
# extensions, i.e., are not subdirs, e.g., list.txt
train_dir_names = \
    list(set(glob.glob(os.path.join(path, "train", "*"))).
         difference(set(glob.glob(os.path.join(path, "train", "*.*")))))
train_dir_names.sort()


def getLargestRegion(props, labelmap, im_thresh):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(im_thresh[labelmap == regionprop.label])*1.0/regionprop.area < \
                0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop



# In Tutorial [9]:
"""
Now, we collect the previous steps together in a function to make it easily 
repeatable.
"""

# Adapted from Tutorial [9]:
def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    im_thresh = np.where(image > np.mean(image), 0., 1.0)

    # Dilate the image
    im_dilated = morphology.dilation(im_thresh, np.ones((4, 4)))

    # Create the label list
    label_list = measure.label(im_dilated)
    label_list = im_thresh*label_list
    label_list = label_list.astype(int)

    regionprops_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(regionprops_list, label_list, im_thresh)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else maxregion.minor_axis_length*1.0\
            / maxregion.major_axis_length
    return ratio


"""
Preparing Training Data

With our code for the ratio of minor to major axis, let's add the raw pixel
values to the list of features for our dataset. In order to use the pixel
values in a model for our classifier, we need a fixed length feature vector, so
we will rescale the images to be constant size and add the fixed number of
pixels to the feature vector.

To create the feature vectors, we will loop through each of the directories in
our training data set and then loop over each image within that class. For each
image, we will rescale it to 25 x 25 pixels and then add the rescaled pixel
values to a feature vector, X. The last feature we include will be our
width-to-length ratio. We will also create the class label in the vector y,
which will have the true class label for each row of the feature vector, X.
"""

# Adapted from Tutorial [10]
# Rescale the images and create the combined metrics and training labels

# get the total training images
numberofImages = 0
for folder in train_dir_names:
    for fileNameDir in os.walk(folder):
        # fileNameDir will be a 3-tuple, (dirpath, dirnames, filenames)
        # so we look at the last element, a list of the filenames
        # print fileNameDir
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue
            numberofImages += 1


# We'll rescale the images to be 25x25=625
# Why 25? Why not 2**5 = 32?
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages  # one row for each image in the training dataset
num_features = imageSize + 1  # for our ratio

# X is the ARRAY of feature vectors with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label
# TODO why the double parens?
y = np.zeros((num_rows))

files = []
# Generate training data
i = 0
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in train_dir_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    print currentClass
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue

            # Read in the images and create the features
            nameFileImage = \
                "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            axisratio = getMinorMajorRatio(image)
# TODO: check out exactly how skimage resizes
            image = resize(image, (maxPixel, maxPixel))

            # Store the rescaled image pixels and the axis ratio
            X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
            X[i, imageSize] = axisratio

            # Store the classlabel
            y[i] = label
            i += 1
            # report progress for each 5% done
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report:
                print np.ceil(i * 100.0 / num_rows), "% done"
    label += 1
    
"""
Width-to-Length Ratio Class Separation

Now that we have calculated the width-to-length ratio metric for all the
images, we can look at the class separation to see how well our feature
performs. We'll compare pairs of the classes' distributions by plotting each
pair of classes. While this will not cover the whole space of hundreds of
possible combinations, it will give us a feel for how similar or dissimilar
different classes are in this feature, and the class distributions should be
comparable across subplots.
"""

# From Tutorial [12]
# Loop through the classes two at a time and compare their distributions of
# the Width/Length Ratio

# Create a DataFrame object to make subsetting the data on the class
df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})

f = plt.figure(figsize=(30, 20))

# Suppress zeros and choose a few large classes to better highlight the
# distributions.
# Here "large" means images that have a large ratio of minor to major axis.
df = df.loc[df["ratio"] > 0]
minimumSize = 20
counts = df["class"].value_counts()
largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]

# Loop through 40 of the classes

for j in range(0, 40, 2):
    subfig = plt.subplot(4, 5, j / 2 + 1)
    # Plot the normalized histograms for two classes
    classind1 = largeclasses[j]
    classind2 = largeclasses[j+1]
    n, bins, p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,
                          alpha=0.5, bins=[x*0.01 for x in range(100)],
                          label=namesClasses[classind1].split(os.sep)[-1],
                          normed=1)

    n2, bins, p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,
                           alpha=0.5, bins=bins,
                           label=namesClasses[classind2].split(os.sep)[-1],
                           normed=1)
    subfig.set_ylim([0., 10.])
    plt.legend(loc='upper right')
    plt.xlabel("Width/Length Ratio")


# results = six histograms in 2x3 display
# TODO: this doesn't make sense, printing out 20 graphs on top of each other.
#       Figure out how to display this reasonably.
"""
From the (truncated) figure above, you will see some cases where the classes
are well separated and others were they are not.
NB:
    It is typical that one single
feature will not allow you to completely separate more than thirty distinct
classes. You will need to be creative in coming up with additional metrics to
discriminate between all the classes.
TODO: Figure out how CNN fits into this task.
"""

"""
TODO: Understand this thoroughly.
Random Forest Classification

We choose a random forest model to classify the images. Random forests perform 
well in many classification tasks and have robust default settings. We will 
give a brief description of a random forest model so that you can understand 
its two main free parameters: n_estimators and max_features.

A random forest model is an ensemble model of n_estimators number of decision 
trees. During the training process, each decision tree is grown automatically 
by making a series of conditional splits on the data. At each split in the 
decision tree, a random sample of max_features number of features is chosen and 
used to make a conditional decision on which of the two nodes that the data 
will be grouped in. The best condition for the split is determined by the split 
that maximizes the class purity of the nodes directly below. The tree continues 
to grow by making additional splits until the leaves are pure or the leaves 
have less than the minimum number of samples for a split (in sklearn default 
for min_samples_split is two data points). The final majority class purity of 
the terminal nodes of the decision tree are used for making predictions on what 
class a new data point will belong. Then, the aggregate vote across the forest 
determines the class prediction for new samples.

With our training data consisting of the feature vector X and the class label 
vector y, we will now calculate some class metrics for the performance of our 
model, by class and overall. First, we train the random forest on all the 
available data and let it perform the 5-fold cross validation. Then we perform 
the cross validation using the KFold method, which splits the data into train 
and test sets, and a classification report. The classification report provides 
a useful list of performance metrics for your classifier vs. the internal 
metrics of the random forest module.
"""

# From Tutorial [19]
print "Training"
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square
# root of the number of features
clf = RF(n_estimators=100, n_jobs=3);
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1);
print "Accuracy of all classes"
print np.mean(scores)


"""
Tutorial Results:
Training
Accuracy of all classes
0.446073202468

# 2/?/2015
I got *very* close:
Accuracy of all classes
0.466980629201

# 2/21/2015 6:50pm Also very close
Training
Accuracy of all classes
0.466064989056

# 2/22/2015
Training
Accuracy of all classes
0.465496298508

"""

# From Tutorial [14]:
# TODO: Understand completely:
#   sklearn.cross_validation import StratifiedKFold as KFold, including results
kf = KFold(y, n_folds=5)
y_pred = y * 0
for train, test in kf:
    X_train, X_test, y_train, y_test=X[train, :], X[test, :], y[train], y[test]
    clf = RF(n_estimators=100, n_jobs=3)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict(X_test)
print classification_report(y, y_pred, target_names=namesClasses)


"""
The current model, while somewhat accurate overall, doesn't do well for all 
classes, including the shrimp caridean, stomatopod, or hydromedusae tentacles 
classes. For others it does quite well, getting many of the correct 
classifications for trichodesmium_puff and copepod_oithona_eggs classes. The 
metrics shown above for measuring model performance include precision, recall, 
and f1-score. 
The precision metric gives probability that a chosen class is correct,
    (true positives / (true positive + false positives)), 
while recall measures the ability of the model to correctly classify examples
of a given class, 
    (true positives / (false negatives + true positives)). 
The F1 score is the geometric average of the precision and recall (the sqrt of
their product).

The competition scoring uses a multiclass log-loss metric to compute your
overall score. In the next steps, we define the multiclass log-loss function
and compute your estimated score on the training dataset.
"""


# From tutorial [16]:
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


# From tutor [17]:
# Get the probability predictions for computing the log-loss function
kf = KFold(y, n_folds=5)
# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(y), len(set(y))))
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    clf = RF(n_estimators=100, n_jobs=3)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict_proba(X_test)


# From tutorial [18]:
multiclass_log_loss(y, y_pred)
"""
Tutorial Results:  3.7390475458333374
My results - very close:
2/?/2015   3.7285067867109327
2/22/2015  3.7570415769375152

"""
""""
The multiclass log loss function is a classification error metric that heavily
penalizes you for being both confident (either predicting very high or very low
class probability) and wrong. Throughout the competition you will want to check
that your model improvements are driving this loss metric lower.
"""

"""
Where to Go From Here

Now that you've made a simple metric, created a model, and examined the model's
performance on the training data, the next step is to make improvements to your
model to make it more competitive. The random forest model we created does not
perform evenly across all classes and in some cases fails completely. By
creating new features and looking at some of your distributions for the problem
classes directly, you can identify features that specifically help separate
those classes from the others. You can add new metrics by considering other
image properties, stratified sampling, transformations, or other models for the
classification.
"""
