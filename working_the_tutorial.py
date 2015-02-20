# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:31:42 2015
This is lifting the kaggle tutorial for the National Data Science Bowl at
https://www.kaggle.com/c/datasciencebowl/details/tutorial
Any code section lifted from the tutorial will start with # In tutorial [n].
My adaption will start with # Adapted
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

# In tutorial [3]:
"""
# get the classnames from the directory structure
directory_names = list(
    set(glob.glob(os.path.join("train", "*")))
    .difference(set(glob.glob(os.path.join("train", "*.*")))))
"""

# Adapted, calling the training dir names train_dir_names.
# NB: glob will NOT expand the tilde! 2 hours of my life before I saw this:
#    https://docs.python.org/2/library/glob.html

# Kathleen:  I commented this out becusae I couldn't figure out the purpose.  I reinserted the \
# original commented code above here to pick up the class names.

directory_names = list(
    set(glob.glob(os.path.join("train", "*")))
    .difference(set(glob.glob(os.path.join("train", "*.*")))))

directory_names.sort()

#
# path = "./"
# path = os.path.abspath(path)
# # Using .differences(...) removes any files in the dir train that have,
# # extensions, i.e., are not subdirs, e.g., list.txt
# train_full_path = \
#     list(set(glob.glob(os.path.join(path, "data", "train", "*"))).
#         difference(set(glob.glob(os.path.join(path, "data", "train", "*.*")))))
# train_full_path.sort()
#
# # Remove path from the dir names
# train_dir_names = []
# train_dir_names = [dn.split('/')[-1] for dn in train_full_path]

# In tutorial [4]:
"""
# We will develop our feature on one image example and examine each step before
# calculating the feature across the distribution of classes.
# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
example_file = glob.glob(os.path.join(directory_names[5], "*.jpg"))[9]
print example_file
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
plt.show()
# competition_data/train/acantharia_protist/101574.jpg
"""


# Adapted from Tutorial [4]:
# This image is not the 8th in the 4th dir for me, so I'll just hardcode it.
# Get a filename

example_file = glob.glob(os.path.join(directory_names[0], "101574.jpg"))[0]
print example_file

# Create an image object by reading the file
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
plt.show()

# What does this plt.show() do?  The plt.imshow already displayed it and this
# doesn't do anything more.
# DONE: TODO: Look it up in matplotlib.
"""
# From http://matplotlib.org/api/pyplot_summary.html
# imshow:	Display an image on the axes.
# show:	Display a figure.

# From http://stackoverflow.com/questions/3497578/matplotlib-plot-and-imshow
plt.show() displays the figure (and enters the mainloop of whatever gui backend
you're using). You shouldn't call it until you've plotted things and want to
see them displayed.

plt.imshow() draws an image on the current figure (creating a figure is [sic]
there isn't a current figure). Calling plt.show() before you've drawn anything
doesn't make any sense. If you want to explictly create a new figure, use
plt.figure().
"""

"""
From the tutorial:
Preparing the Images

To create the features of interest, we will need to prepare the images by
completing a few preprocessing procedures. We will step through some common
image preprocessing actions: thresholding the images, segmenting the images,
and extracting region properties. Using the region properties, we will create
features based on the intrinsic properties of the classes, which we expect will
allow us discriminate between them. We walk through the process of adding one
such feature for the ratio of the width by length of the object of interest.

First, we begin by thresholding the image on the the mean value. This will
reduce some of the noise in the image. Then, we apply a three step segmentation
process: first we dilate the image to connect neighboring pixels, then we
calculate the labels for connected regions, and finally we apply the original
threshold to the labels to label the original, undilated regions.
"""

# In tutorial [5]:
# First we threshold the image by only taking values greater than the mean to
# reduce noise in the image to use later as a mask
"""
f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")

sub2 = plt.subplot(1,4,2)
plt.imshow(imthr, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
sub3 = plt.subplot(1, 4, 3)
plt.imshow(imdilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

labels = measure.label(imdilated)
labels = imthr*labels
labels = labels.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(labels)
"""

# Adapted:
# How did they know to use this figure size?  Oh, ok, it's just a panel that's
# big enough for a few (3? - no, 4) little images.
# TODO: study pyplot: is f just the current figure? Can there be >1?  How do
# you kill it?
f = plt.figure(figsize=(12, 3))
im_thresholded = im.copy()
im_thresholded = np.where(im > np.mean(im), 0., 1.0)
sub1 = plt.subplot(1, 4, 1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")
# subtitle is not showing up yet on panel ...

sub2 = plt.subplot(1, 4, 2)
plt.imshow(im_thresholded, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

im_dilated = morphology.dilation(im_thresholded, np.ones((4, 4)))
# DONE TODO: check out morphology from skimage
"""
http://scikit-image.org/docs/dev/api/
    skimage.morphology.html?highlight=morphology#skimage.morphology.dilation
Morphological dilation sets a pixel at (i,j) to the maximum over all pixels in
the neighborhood centered at (i,j). Dilation enlarges bright regions and
shrinks dark regions.

Here, the image is 1s and 0s, so each 1 is extended to the fill the 4x4 region
beyond itself.
"""

sub3 = plt.subplot(1, 4, 3)
plt.imshow(im_dilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

# DONE TODO: figure out what's going on here. How does measure from skimage work?'
"""
http://scikit-image.org/docs/dev/api/
    skimage.measure.html?highlight=measure#label
Very cool: measure.label labels any set of neighboring (4- or 8-neighbor)
pixels that have the same value as a region.
"""
im_labeled = measure.label(im_dilated)
# This multiplication will shrink the labeled image back to the
# original, undilated image pixels.  It changes it to float64.
im_labeled = im_thresholded*im_labeled
# numpy.ndarray.astype: Copy of the array, cast to a specified type. After this
# it's back to int64.
im_labeled = im_labeled.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(im_labeled)


# TODO
# Up to here, all works fine except that the panel is displayed inline on the
# console, while it should be in a separate panel. I changed the Spyder
# graphic preferences to "Automatic" but it hasn't been applied.  Maybe on next
# startup.


# In tutorial [6]:
"""
With the image segmented into different parts, we would like to choose the
largest non-background part to compute our metric. We would like to select the
largest segment as the likely object of interest for classification purposes.
We loop through the available regions and select the one with the largest area.
There are many properties available within the regions that you can explore for
creating new features. Look at the documentation for regionprops for
inspiration.
"""
# Adapted:
# See measure.regionprops here: http://scikit-image.org/docs/dev/
#    search.html?q=regionprops&check_keywords=yes&area=default#
# Calculate common region properties for each region within the segmentation
# (For hard-coded image, acantharia_protist/101574.jpg, there are 4 regions.)
# The var regions is actually the properties for the image's regions, not the 
# regions themselves.
regions = measure.regionprops(im_labeled)

# find the largest nonzero region
def getLargestRegion(props=regions, labelmap=im_labeled,
                     im_thresh=im_thresholded):
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


"""
The results for our test image are shown below. The segmentation picked one
region and we use that region to calculate our ratio metric.
"""
# Adapted from tutorial [7]:
# If no regions are fed to getLargestRegion, it vars named regions, im_labeled
# and im_thresholded.
regionmax = getLargestRegion()
plt.imshow(np.where(im_labeled == regionmax.label, 1.0, 0.0))
plt.show()


# Adapted from tutorial [8]:
print 'regionmax.label = %d' % regionmax.label
print 'regionmax axis ratio = %f' % \
    (regionmax.minor_axis_length/regionmax.major_axis_length)

# TODO Tutorial gets 0.144141699631; I get 0.687475491704. Hmm.. Figure it out.
# Start by looking at this ratio, and the eccentricity, for all 4 regions:
for region in regions:
    print 'region %d: eccentriciy = %f' % (region.label, region.eccentricity)
    print '          minor to major axis = %f' %\
          (region.minor_axis_length/region.major_axis_length)
# The ratio .144 seems too low. If an ellipse were fitted to regionmax it would
# not be that flat.  I think it's a typo in Tutorial.py comments.
plt.imshow(np.where(im_labeled == regionmax.label, 1.0, 0.0))


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
for folder in train_full_path:
    for fileNameDir in os.walk(folder):
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
for folder in train_full_path:
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
# TODO: figure out exactly what this means "{0}{1}{2}" = ?, os.sep = "/"?
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
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
    

"""
# results = six histograms in 2x3 display
# TODO: this doesn't make sense, printing out 20 graphs on top of each other.
#       Figure out how to display this reasonably.
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

I got *very* close:
Accuracy of all classes
0.466980629201

"""

# From Tutorial [14]:
# TODO: Understand completely:
#   sklearn.cross_validation import StratifiedKFold as KFold, including results
#   DUH.
kf = KFold(y, n_folds=5)
y_pred = y * 0
for train, test in kf:
    X_train, X_test, y_train, y_test=X[train, :], X[test, :], y[train], y[test]
    clf = RF(n_estimators=100, n_jobs=3)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict(X_test)
print classification_report(y, y_pred, target_names=namesClasses)

"""
Tutorial Resluts:
precision recall f1-score support

competition_data/train/appendicularian_slight_curve 0.32 0.43 0.37 532
competition_data/train/hydromedusae_shapeB 0.33 0.08 0.13 150
competition_data/train/hydromedusae_shapeA 0.44 0.80 0.57 412
competition_data/train/siphonophore_other_parts 0.00 0.00 0.00 29
competition_data/train/tunicate_doliolid_nurse 0.24 0.07 0.11 417
competition_data/train/acantharia_protist 0.40 0.85 0.54 889
competition_data/train/hydromedusae_narco_young 0.23 0.08 0.12 336
competition_data/train/fish_larvae_deep_body 0.00 0.00 0.00 10
competition_data/train/hydromedusae_haliscera_small_sideview 0.00 0.00 0.00 9
competition_data/train/echinoderm_larva_pluteus_early 0.46 0.26 0.33 92
competition_data/train/siphonophore_calycophoran_rocketship_young 0.33 0.23 0.27 483
competition_data/train/chaetognath_sagitta 0.38 0.20 0.26 694
competition_data/train/shrimp-like_other 0.00 0.00 0.00 52
competition_data/train/hydromedusae_liriope 0.00 0.00 0.00 19
competition_data/train/heteropod 0.00 0.00 0.00 10
competition_data/train/appendicularian_straight 0.50 0.04 0.07 242
competition_data/train/chaetognath_non_sagitta 0.57 0.74 0.65 815
competition_data/train/trochophore_larvae 0.00 0.00 0.00 29
competition_data/train/copepod_cyclopoid_oithona_eggs 0.54 0.80 0.64 1189
competition_data/train/ctenophore_cestid 0.00 0.00 0.00 113
competition_data/train/pteropod_butterfly 0.60 0.08 0.15 108
competition_data/train/hydromedusae_sideview_big 0.00 0.00 0.00 76
competition_data/train/siphonophore_calycophoran_rocketship_adult 0.36 0.04 0.07 135
competition_data/train/shrimp_caridean 0.68 0.27 0.38 49
competition_data/train/hydromedusae_typeD 0.00 0.00 0.00 43
competition_data/train/appendicularian_s_shape 0.37 0.54 0.44 696
competition_data/train/crustacean_other 0.25 0.11 0.15 201
competition_data/train/fish_larvae_myctophids 0.55 0.54 0.54 114
competition_data/train/hydromedusae_partial_dark 0.66 0.27 0.38 190
competition_data/train/copepod_calanoid_eggs 0.80 0.19 0.31 173
competition_data/train/stomatopod 0.00 0.00 0.00 24
competition_data/train/siphonophore_physonect_young 0.00 0.00 0.00 21
competition_data/train/hydromedusae_solmundella 0.77 0.16 0.27 123
competition_data/train/ephyra 0.00 0.00 0.00 14
competition_data/train/hydromedusae_bell_and_tentacles 0.00 0.00 0.00 75
competition_data/train/pteropod_triangle 0.00 0.00 0.00 65
competition_data/train/hydromedusae_h15 0.47 0.20 0.28 35
competition_data/train/diatom_chain_string 0.55 0.91 0.69 519
competition_data/train/hydromedusae_narcomedusae 0.00 0.00 0.00 132
competition_data/train/copepod_calanoid_large 0.48 0.41 0.44 286
competition_data/train/radiolarian_colony 0.44 0.23 0.30 158
competition_data/train/tunicate_partial 0.61 0.95 0.75 352
competition_data/train/invertebrate_larvae_other_B 0.00 0.00 0.00 24
competition_data/train/invertebrate_larvae_other_A 0.00 0.00 0.00 14
competition_data/train/echinoderm_larva_pluteus_brittlestar 0.60 0.08 0.15 36
competition_data/train/siphonophore_calycophoran_abylidae 0.21 0.04 0.07 212
competition_data/train/euphausiids_young 0.00 0.00 0.00 38
competition_data/train/hydromedusae_aglaura 0.00 0.00 0.00 127
competition_data/train/protist_dark_center 0.00 0.00 0.00 108
competition_data/train/trichodesmium_bowtie 0.44 0.69 0.54 708
competition_data/train/radiolarian_chain 0.58 0.07 0.12 287
competition_data/train/protist_fuzzy_olive 0.75 0.78 0.76 372
competition_data/train/polychaete 0.75 0.02 0.04 131
competition_data/train/copepod_calanoid 0.39 0.55 0.46 681
competition_data/train/amphipods 0.00 0.00 0.00 49
competition_data/train/acantharia_protist_big_center 0.00 0.00 0.00 13
competition_data/train/copepod_calanoid_octomoms 0.00 0.00 0.00 49
competition_data/train/protist_other 0.36 0.68 0.47 1172
competition_data/train/hydromedusae_other 0.00 0.00 0.00 12
competition_data/train/tunicate_salp 0.59 0.75 0.66 236
competition_data/train/siphonophore_calycophoran_sphaeronectes_stem 0.00 0.00 0.00 57
competition_data/train/trichodesmium_puff 0.71 0.93 0.80 1979
competition_data/train/artifacts 0.54 0.85 0.66 393
competition_data/train/fish_larvae_leptocephali 0.00 0.00 0.00 31
competition_data/train/echinoderm_larva_seastar_bipinnaria 0.54 0.56 0.55 385
competition_data/train/chordate_type1 0.52 0.56 0.54 77
competition_data/train/shrimp_zoea 0.54 0.20 0.29 174
competition_data/train/fish_larvae_very_thin_body 0.00 0.00 0.00 16
competition_data/train/ctenophore_cydippid_no_tentacles 0.00 0.00 0.00 42
competition_data/train/appendicularian_fritillaridae 0.00 0.00 0.00 16
competition_data/train/siphonophore_physonect 0.00 0.00 0.00 128
competition_data/train/trichodesmium_tuft 0.37 0.46 0.41 678
competition_data/train/fecal_pellet 0.32 0.29 0.31 511
competition_data/train/hydromedusae_shapeA_sideview_small 0.37 0.10 0.16 274
competition_data/train/siphonophore_calycophoran_sphaeronectes 0.44 0.11 0.17 179
competition_data/train/artifacts_edge 0.89 0.75 0.81 170
competition_data/train/ctenophore_cydippid_tentacles 0.00 0.00 0.00 53
competition_data/train/copepod_cyclopoid_oithona 0.50 0.61 0.55 899
competition_data/train/siphonophore_partial 0.00 0.00 0.00 30
competition_data/train/tunicate_doliolid 0.26 0.17 0.20 439
competition_data/train/copepod_other 0.00 0.00 0.00 24
competition_data/train/unknown_blobs_and_smudges 0.27 0.13 0.18 317
competition_data/train/shrimp_sergestidae 0.72 0.08 0.15 153
competition_data/train/hydromedusae_solmaris 0.40 0.51 0.44 703
competition_data/train/copepod_calanoid_flatheads 0.50 0.02 0.04 178
competition_data/train/echinoderm_larva_seastar_brachiolaria 0.61 0.76 0.68 536
competition_data/train/copepod_calanoid_eucalanus 0.88 0.07 0.13 96
competition_data/train/ctenophore_lobate 0.74 0.45 0.56 38
competition_data/train/detritus_filamentous 0.12 0.01 0.01 394
competition_data/train/jellies_tentacles 0.40 0.01 0.03 141
competition_data/train/detritus_blob 0.19 0.06 0.09 363
competition_data/train/chaetognath_other 0.43 0.74 0.54 1934
competition_data/train/copepod_cyclopoid_copilia 0.00 0.00 0.00 30
competition_data/train/copepod_calanoid_large_side_antennatucked 0.50 0.23 0.31 106
competition_data/train/trichodesmium_multiple 1.00 0.07 0.14 54
competition_data/train/fish_larvae_thin_body 0.00 0.00 0.00 64
competition_data/train/diatom_chain_tube 0.39 0.41 0.40 500
competition_data/train/tunicate_salp_chains 0.00 0.00 0.00 73
competition_data/train/protist_star 0.84 0.58 0.68 113
competition_data/train/fish_larvae_medium_body 0.49 0.32 0.39 85
competition_data/train/hydromedusae_narco_dark 0.00 0.00 0.00 23
competition_data/train/hydromedusae_haliscera 0.54 0.40 0.46 229
competition_data/train/hydromedusae_typeE 0.00 0.00 0.00 14
competition_data/train/hydromedusae_typeF 0.54 0.21 0.31 61
competition_data/train/echinoderm_larva_pluteus_urchin 0.59 0.26 0.36 88
competition_data/train/siphonophore_calycophoran_sphaeronectes_young 0.38 0.04 0.07 247
competition_data/train/protist_noctiluca 0.57 0.50 0.53 625
competition_data/train/copepod_calanoid_frillyAntennae 1.00 0.03 0.06 63
competition_data/train/echinoderm_seacucumber_auricularia_larva 0.00 0.00 0.00 96
competition_data/train/tornaria_acorn_worm_larvae 0.84 0.42 0.56 38
competition_data/train/detritus_other 0.26 0.33 0.29 914
competition_data/train/unknown_sticks 0.43 0.06 0.10 175
competition_data/train/unknown_unclassified 0.14 0.01 0.02 425
competition_data/train/pteropod_theco_dev_seq 0.00 0.00 0.00 13
competition_data/train/acantharia_protist_halo 1.00 0.03 0.05 71
competition_data/train/hydromedusae_typeD_bell_and_tentacles 0.75 0.11 0.19 56
competition_data/train/echinoderm_larva_pluteus_typeC 0.71 0.15 0.25 80
competition_data/train/decapods 0.00 0.00 0.00 55
competition_data/train/copepod_calanoid_small_longantennae 0.70 0.08 0.14 87
competition_data/train/euphausiids 0.55 0.09 0.15 136
competition_data/train/echinopluteus 0.00 0.00 0.00 27

avg / total 0.43 0.47 0.41 30336
"""

"""
My results:
avg / total       0.44      0.47      0.41     30336
# This is extremely close to what they got.

                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/acantharia_protist       0.40      0.86      0.55       889
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/acantharia_protist       0.00      0.00      0.00        13
                /Users/kmgpl/Documents/DDL_SeeFish/data/train/acantharia_protist_big_center       1.00      0.06      0.11        71
                      /Users/kmgpl/Documents/DDL_SeeFish/data/train/acantharia_protist_halo       0.00      0.00      0.00        49
                                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/amphipods       0.00      0.00      0.00        16
                /Users/kmgpl/Documents/DDL_SeeFish/data/train/appendicularian_fritillaridae       0.35      0.53      0.42       696
                      /Users/kmgpl/Documents/DDL_SeeFish/data/train/appendicularian_s_shape       0.34      0.41      0.37       532
                 /Users/kmgpl/Documents/DDL_SeeFish/data/train/appendicularian_slight_curve       0.50      0.02      0.05       242
                     /Users/kmgpl/Documents/DDL_SeeFish/data/train/appendicularian_straight       0.53      0.84      0.65       393
                                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/artifacts       0.93      0.81      0.86       170
                               /Users/kmgpl/Documents/DDL_SeeFish/data/train/artifacts_edge       0.56      0.73      0.64       815
                      /Users/kmgpl/Documents/DDL_SeeFish/data/train/chaetognath_non_sagitta       0.40      0.76      0.53      1934
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/chaetognath_other       0.44      0.21      0.28       694
                          /Users/kmgpl/Documents/DDL_SeeFish/data/train/chaetognath_sagitta       0.55      0.58      0.57        77
                               /Users/kmgpl/Documents/DDL_SeeFish/data/train/chordate_type1       0.41      0.60      0.48       681
                             /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid       0.74      0.16      0.27       173
                        /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_eggs       1.00      0.10      0.19        96
                   /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_eucalanus       0.50      0.03      0.05       178
                   /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_flatheads       1.00      0.02      0.03        63
              /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_frillyAntennae       0.50      0.42      0.46       286
                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_large       0.45      0.21      0.28       106
    /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_large_side_antennatucked       0.00      0.00      0.00        49
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_octomoms       0.94      0.18      0.31        87
          /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_calanoid_small_longantennae       1.00      0.03      0.06        30
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_cyclopoid_copilia       0.47      0.62      0.53       899
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_cyclopoid_oithona       0.56      0.78      0.66      1189
               /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_cyclopoid_oithona_eggs       0.00      0.00      0.00        24
                                /Users/kmgpl/Documents/DDL_SeeFish/data/train/copepod_other       0.30      0.13      0.18       201
                             /Users/kmgpl/Documents/DDL_SeeFish/data/train/crustacean_other       0.00      0.00      0.00       113
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/ctenophore_cestid       0.00      0.00      0.00        42
             /Users/kmgpl/Documents/DDL_SeeFish/data/train/ctenophore_cydippid_no_tentacles       0.00      0.00      0.00        53
                /Users/kmgpl/Documents/DDL_SeeFish/data/train/ctenophore_cydippid_tentacles       0.70      0.50      0.58        38
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/ctenophore_lobate       0.00      0.00      0.00        55
                                     /Users/kmgpl/Documents/DDL_SeeFish/data/train/decapods       0.16      0.06      0.08       363
                                /Users/kmgpl/Documents/DDL_SeeFish/data/train/detritus_blob       0.13      0.02      0.03       394
                         /Users/kmgpl/Documents/DDL_SeeFish/data/train/detritus_filamentous       0.24      0.37      0.29       914
                               /Users/kmgpl/Documents/DDL_SeeFish/data/train/detritus_other       0.56      0.92      0.69       519
                          /Users/kmgpl/Documents/DDL_SeeFish/data/train/diatom_chain_string       0.38      0.43      0.40       500
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/diatom_chain_tube       0.60      0.08      0.15        36
         /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_larva_pluteus_brittlestar       0.57      0.29      0.39        92
               /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_larva_pluteus_early       0.92      0.15      0.26        80
               /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_larva_pluteus_typeC       0.67      0.35      0.46        88
              /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_larva_pluteus_urchin       0.57      0.59      0.58       385
          /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_larva_seastar_bipinnaria       0.61      0.80      0.69       536
        /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_larva_seastar_brachiolaria       0.00      0.00      0.00        96
     /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinoderm_seacucumber_auricularia_larva       0.00      0.00      0.00        27
                                /Users/kmgpl/Documents/DDL_SeeFish/data/train/echinopluteus       0.00      0.00      0.00        14
                                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/ephyra       0.48      0.07      0.13       136
                                  /Users/kmgpl/Documents/DDL_SeeFish/data/train/euphausiids       0.33      0.03      0.05        38
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/euphausiids_young       0.33      0.29      0.31       511
                                 /Users/kmgpl/Documents/DDL_SeeFish/data/train/fecal_pellet       0.00      0.00      0.00        10
                        /Users/kmgpl/Documents/DDL_SeeFish/data/train/fish_larvae_deep_body       0.00      0.00      0.00        31
                     /Users/kmgpl/Documents/DDL_SeeFish/data/train/fish_larvae_leptocephali       0.51      0.32      0.39        85
                      /Users/kmgpl/Documents/DDL_SeeFish/data/train/fish_larvae_medium_body       0.56      0.58      0.57       114
                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/fish_larvae_myctophids       0.33      0.02      0.03        64
                        /Users/kmgpl/Documents/DDL_SeeFish/data/train/fish_larvae_thin_body       0.00      0.00      0.00        16
                   /Users/kmgpl/Documents/DDL_SeeFish/data/train/fish_larvae_very_thin_body       0.00      0.00      0.00        10
                                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/heteropod       0.00      0.00      0.00       127
                         /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_aglaura       0.33      0.03      0.05        75
              /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_bell_and_tentacles       0.47      0.23      0.31        35
                             /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_h15       0.49      0.41      0.45       229
                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_haliscera       0.00      0.00      0.00         9
        /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_haliscera_small_sideview       0.00      0.00      0.00        19
                         /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_liriope       0.00      0.00      0.00        23
                      /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_narco_dark       0.38      0.08      0.13       336
                     /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_narco_young       0.00      0.00      0.00       132
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_narcomedusae       0.00      0.00      0.00        12
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_other       0.57      0.21      0.31       190
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_partial_dark       0.43      0.74      0.54       412
                          /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_shapeA       0.35      0.09      0.15       274
           /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_shapeA_sideview_small       0.37      0.07      0.12       150
                          /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_shapeB       0.00      0.00      0.00        76
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_sideview_big       0.41      0.51      0.45       703
                        /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_solmaris       0.92      0.19      0.31       123
                     /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_solmundella       0.00      0.00      0.00        43
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_typeD       0.75      0.05      0.10        56
        /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_typeD_bell_and_tentacles       0.00      0.00      0.00        14
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_typeE       0.52      0.21      0.30        61
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/hydromedusae_typeF       0.00      0.00      0.00        14
                  /Users/kmgpl/Documents/DDL_SeeFish/data/train/invertebrate_larvae_other_A       0.00      0.00      0.00        24
                  /Users/kmgpl/Documents/DDL_SeeFish/data/train/invertebrate_larvae_other_B       0.55      0.04      0.08       141
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/jellies_tentacles       0.00      0.00      0.00       131
                                   /Users/kmgpl/Documents/DDL_SeeFish/data/train/polychaete       0.00      0.00      0.00       108
                          /Users/kmgpl/Documents/DDL_SeeFish/data/train/protist_dark_center       0.74      0.78      0.76       372
                          /Users/kmgpl/Documents/DDL_SeeFish/data/train/protist_fuzzy_olive       0.56      0.54      0.55       625
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/protist_noctiluca       0.37      0.68      0.48      1172
                                /Users/kmgpl/Documents/DDL_SeeFish/data/train/protist_other       0.86      0.57      0.68       113
                                 /Users/kmgpl/Documents/DDL_SeeFish/data/train/protist_star       0.86      0.06      0.10       108
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/pteropod_butterfly       0.00      0.00      0.00        13
                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/pteropod_theco_dev_seq       0.50      0.02      0.03        65
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/pteropod_triangle       0.38      0.03      0.06       287
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/radiolarian_chain       0.46      0.23      0.31       158
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/radiolarian_colony       0.00      0.00      0.00        52
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/shrimp-like_other       0.65      0.22      0.33        49
                              /Users/kmgpl/Documents/DDL_SeeFish/data/train/shrimp_caridean       0.67      0.07      0.12       153
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/shrimp_sergestidae       0.51      0.20      0.29       174
                                  /Users/kmgpl/Documents/DDL_SeeFish/data/train/shrimp_zoea       0.23      0.03      0.06       212
           /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_calycophoran_abylidae       0.50      0.07      0.12       135
   /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_calycophoran_rocketship_adult       0.37      0.22      0.28       483
   /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_calycophoran_rocketship_young       0.50      0.13      0.21       179
      /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_calycophoran_sphaeronectes       0.00      0.00      0.00        57
 /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_calycophoran_sphaeronectes_stem       0.26      0.04      0.06       247
/Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_calycophoran_sphaeronectes_young       0.00      0.00      0.00        29
                     /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_other_parts       0.00      0.00      0.00        30
                         /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_partial       0.00      0.00      0.00       128
                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_physonect       0.00      0.00      0.00        21
                 /Users/kmgpl/Documents/DDL_SeeFish/data/train/siphonophore_physonect_young       0.00      0.00      0.00        24
                                   /Users/kmgpl/Documents/DDL_SeeFish/data/train/stomatopod       0.79      0.39      0.53        38
                   /Users/kmgpl/Documents/DDL_SeeFish/data/train/tornaria_acorn_worm_larvae       0.46      0.65      0.54       708
                         /Users/kmgpl/Documents/DDL_SeeFish/data/train/trichodesmium_bowtie       1.00      0.13      0.23        54
                       /Users/kmgpl/Documents/DDL_SeeFish/data/train/trichodesmium_multiple       0.73      0.92      0.81      1979
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/trichodesmium_puff       0.37      0.44      0.40       678
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/trichodesmium_tuft       0.00      0.00      0.00        29
                           /Users/kmgpl/Documents/DDL_SeeFish/data/train/trochophore_larvae       0.27      0.18      0.22       439
                            /Users/kmgpl/Documents/DDL_SeeFish/data/train/tunicate_doliolid       0.25      0.05      0.08       417
                      /Users/kmgpl/Documents/DDL_SeeFish/data/train/tunicate_doliolid_nurse       0.62      0.96      0.76       352
                             /Users/kmgpl/Documents/DDL_SeeFish/data/train/tunicate_partial       0.60      0.73      0.66       236
                                /Users/kmgpl/Documents/DDL_SeeFish/data/train/tunicate_salp       0.50      0.01      0.03        73
                         /Users/kmgpl/Documents/DDL_SeeFish/data/train/tunicate_salp_chains       0.26      0.14      0.18       317
                    /Users/kmgpl/Documents/DDL_SeeFish/data/train/unknown_blobs_and_smudges       0.43      0.05      0.09       175
                               /Users/kmgpl/Documents/DDL_SeeFish/data/train/unknown_sticks       0.19      0.01      0.02       425

                                                                                avg / total       0.44      0.47      0.41     30336
"""


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


"""
"""
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
Tutorial Results:
3.7390475458333374
My results - very close:
3.7285067867109327
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
