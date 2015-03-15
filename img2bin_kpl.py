# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 22:08:44 2015
Adapted from 
https://github.com/antinucleon/cxxnet/blob/master/example/kaggle_bowl/gen_img_list.py
This creates binary images from the originals resized to 48x48.  

@author: kperez-lopez
"""
# import csv
import os

# Grabbed all these from the tutorial. Cull it.
from skimage.io import imread
from skimage.io import imsave
import numpy as np

"""
NOTE: I spent a lot of time haggling with these dirs.Seems "resized" need //
    so the /r must be an escape sequence?  Also, it didn't like the 02 in 
    the git dir, 02-SeeFish. Anyway, it was good to move Data dir out of the 
    git dir.
"""
# This is horribly kludgy; will slick it up later.
# parameters to read the list of training images
params = ["train", 
          "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\"]

# parameters to read the list of test images
params = ["test", 
          "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\"]

task = params[0]
    
fi = params[1] + "Data\\resized\\" + task 
fo = params[1] + "Data\\bin_imgs\\" + task

items = os.listdir(fi)
print(items[:2])

os.chdir(fo)

count = 0
if task == "train":
    for cls in items:
        try:
            os.mkdir(cls)
        except:
            pass    
        imgs = os.listdir(fi + "\\" + cls)        
        for img in imgs:
            infilename = fi + "\\" + cls + "\\" + img
            im = imread(infilename, as_grey=True)
            im_thresholded = np.where(im > np.mean(im),1.0,0.0)
            outfilename = fo + "\\" + cls + "\\" + img
            if count % 11 == 1:
                print(outfilename)
            imsave(outfilename, im_thresholded)
            count += 1 
else:
    if task == "test":     
        for img in items:
            infilename = fi + "\\" + img
            im = imread(infilename, as_grey=True)
            im_thresholded = np.where(im > np.mean(im),1.0,0.0)
            outfilename = fo + "\\" + img
            if count % 3203 == 1:
                print(outfilename)
            imsave(outfilename, im_thresholded)
            count += 1
    else:
        print("Task is " + task + "; it can only be train or test")
        break