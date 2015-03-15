# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 22:08:44 2015
Adapted from 
https://github.com/antinucleon/cxxnet/blob/master/example/kaggle_bowl/gen_img_list.py

@author: kperez-lopez
"""
import csv
import os
import sys
import random

# This is horribly kludgy; will slick it up later.
# parameters to create the list of training images
params = ["train", "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\02-seefish\\",
          "Data\submission\sampleSubmission.csv",
          "Data\\resized\\train\\",
          "train.lst"]

# parameters to create the list of test images
params = ["test", "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\02-seefish\\",
          "Data\submission\sampleSubmission.csv",
          "Data\\resized\\test\\",
          "test.lst"]
    
os.chdir(params[1])

random.seed(888)

task = params[0]
fc = csv.reader(file(params[2]))
fi = params[3]
fo = csv.writer(open(params[4], "w"), delimiter='\t', lineterminator='\n')


# make class map
head = fc.next()
# skip the first element, which is the column header, "image"
head = head[1:]


# make image list
img_lst = []
cnt = 0
if task == "train":
    for i in xrange(len(head)):
        path = fi + head[i]
        lst = os.listdir(fi + head[i])
        for img in lst:
            img_lst.append((cnt, i, path + '\\' + img))
            cnt += 1
else:
    lst = os.listdir(fi)
    for img in lst:
        img_lst.append((cnt, 0, fi + img))
        cnt += 1

# shuffle
random.shuffle(img_lst)

#write
for item in img_lst:
    fo.writerow(item)
    
# -----------------------------------------
# Original Kaggle site program

if len(sys.argv) < 4:
    print "Usage: gen_img_list.py train/test sample_submission.csv train_folder img.lst"
    exit(1)

random.seed(888)

task = sys.argv[1]
fc = csv.reader(file(sys.argv[2]))
fi = sys.argv[3]
fo = csv.writer(open(sys.argv[4], "w"), delimiter='\t', lineterminator='\n')

# make class map
head = fc.next()
head = head[1:]

# make image list
img_lst = []
cnt = 0
if task == "train":
    for i in xrange(len(head)):
        path = fi + head[i]
        lst = os.listdir(fi + head[i])
        for img in lst:
            img_lst.append((cnt, i, path + '/' + img))
            cnt += 1
else:
    lst = os.listdir(fi)
    for img in lst:
        img_lst.append((cnt, 0, fi + img))
        cnt += 1

# shuffle
random.shuffle(img_lst)

#wirte
for item in img_lst:
    fo.writerow(item)
    
    
