# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 21:56:05 2015
Adapted from 
https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl
gen_test.py

I use PILLOW; original uses ImageMagick, "convert -resize 48x48\!"  The "!"
has to do with aspect ratio, preserving it or not. 
TODO: Figure out which, and then whether I’m doing the same.

This test version is very similar to the image resizing done in get_train.py, 
except that there are no subfolders.  In the training case, the images were 
kept in an identifying folder, here we don't know the identity of the images.

If we don't need the 8-bit 48x48 images, it would make sense to do the resizing
and converting to binary in one sweep.
TODO: Check whether the 8-bit versions of the resized images are ever used. If
the resizing needs to be redone because of aspect ratio, and the 8-bit imgs are
never used, combine the operations in one routine.

TODO: Rework this as it was originally, a program to be called with input args.

@author: kperez-lopez
"""
import os
from PIL import Image
#from __future__ import print_function

"""
# More generally, I should use this:
if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]
"""


fi = "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\02-seefish\Data\original\\test\\"
fo = "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\02-seefish\Data\resized\\test\\"

os.chdir(fo)

small_size = (48, 48)

imgs = os.listdir(fi)

# max_try = 4
# num_tries = 0
    
for img in imgs:
#    if num_tries < max_try:
    im = Image.open(fi + img) 
    small_im = im.resize(small_size)
    small_im.save(fo + img,"JPEG")
    num_tries += 1         
        
# -----------------------------------------------------
# Original file get_train.py from the Kaggle repo:               
import os
import sys
import subprocess

"""
if len(sys.argv) < 3:
    print "Usage: python gen_test.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]
"""

# These folders have been changed back to our git repo.
fi = u"C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\Data\\test\\"
fo = "C:\Users\kperez-lopez\Docments\DataScience\DistrictDataLabs\KaggleSeeFish\cxxnet\example\kaggle_bowl\data\\test\\"

cmd = "convert -resize 48x48\! "
imgs = os.listdir(fi)

for img in imgs:
    md = ""
    md += cmd
    md += fi + img
    md += " " + fo + img
    os.system(md)