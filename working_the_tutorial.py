# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:31:42 2015
This is lifting the kaggle tutorial for the National Data Science Bowl at
https://www.kaggle.com/c/datasciencebowl/details/tutorial
Any code section lifted from the tutorial will start with #In [n].  My
adaption will start with # Adapted
@author: kperez-lopez
"""
#In [1]:
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
%matplotlib inline

#In [2]:  ( don't know why they include this)
import warnings
warnings.filterwarnings("ignore")

#In [3]:
# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
 
# Adapted, calling the training dir names tr_dir_names.
# This removes any files in the dir train that have extensions, i.e., are not
# subdirs, e.g., list.txt
 path = \
  "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\Data\\"
  
tr_dir_names_full = list(set(glob.glob(os.path.join(path,"train", "*"))\
 ).difference(set(glob.glob(os.path.join(path,"train","*.*")))))

tr_dir_names_full.sort()
# Remove my path from the dir names
tr_dir_names = [dn.split('\\')[-1] for dn in tr_dn ]

#In [4]:
# We will develop our feature on one image example and examine each step before 
# calculating the feature across the distribution of classes.
# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
example_file = glob.glob(os.path.join(directory_names[5],"*.jpg"))[9]
print example_file
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
plt.show()
#competition_data/train/acantharia_protist/101574.jpg

# Adapted:
# This image is not the 8th in the 4th dir for me, so I'll just hardcode it.
example_file = glob.glob(os.path.join(tr_dir_names_full[0],"101574.jpg"))[0]
print example_file
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
# What does this plt.show() do?  The plt.imshow already displayed it and this 
# doesn't do anything more.  
# TODO: Look it up in matplotlib.
plt.show()

"""
Preparing the Images

To create the features of interest, we will need to prepare the images by 
completing a few preprocessing procedures. We will step through some common 
image preprocessing actions: thresholding the images, segmenting the images, 
and extracting region properties. Using the region properties, we will create 
features based on the intrinsic properties of the classes, which we expect will
allow us discriminate between them. Let's walk through the process of adding 
one such feature for the ratio of the width by length of the object of interest.

First, we begin by thresholding the image on the the mean value. This will 
reduce some of the noise in the image. Then, we apply a three step segmentation
process: first we dilate the image to connect neighboring pixels, then we 
calculate the labels for connected regions, and finally we apply the original 
threshold to the labels to label the original, undilated regions.
"""

#In [5]: 
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

#Adapted:
# How did they know to use this figure size?  Oh, ok, it's just a panel that's
# big enough for a few (3? - no, 4) little images.
# TODO: study pyplot: is f just the current figure? Can there be >1?  How do
# you kill it?
f = plt.figure(figsize=(12,3))
im_thresholded = im.copy()
im_thresholded = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image") 
# subtitle is not showing up yet on panel ...

sub2 = plt.subplot(1,4,2)
plt.imshow(im_thresholded, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

im_dilated = morphology.dilation(im_thresholded, np.ones((4,4)))
# Does this just apply the averaging filter (4x4 grid of 1s)?
# TODO: check out morphology from skimage
sub3 = plt.subplot(1, 4, 3)
plt.imshow(im_dilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")


# TODO: figure out what's going on here. How does measure from skimage work?
im_labeled = measure.label(im_dilated)
im_labeled = im_thresholded*im_labeled
im_labeled = labeled.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(im_labeled)
