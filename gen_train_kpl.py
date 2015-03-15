# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 22:08:44 2015
Adapted from 
https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl
gen_train.py

I use PILLOW; original seems to be using ImageMagick.
TODO: Figure out which, and then whether I’m doing the same.

This train version is very similar to the image resizing done in get_train.py, 
except that it has all the identifying subfolders. In the test case, the images
are kept in one folder, as we don't know their identities.

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
fi = "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\02-seefish\\train\\"
fo = "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\cxxnet\example\kaggle_bowl\data\\train\\"

fo = "C:\Users\kperez-lopez\Documents\DataScience\DistrictDataLabs\KaggleSeeFish\\02-seefish\Data\resized\\train\\"

os.chdir(fo)

small_size = (48, 48)

classes = os.listdir(fi)

for cls in classes:
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(fi + cls)        
    for img in imgs:
        infilename = fi + cls + "\\" + img
        im = Image.open(infilename) 
        small_im = im.resize(small_size)
        outfilename = fo + cls + "\\" + img
        small_im.save(outfilename,"JPEG")

# -----------------------------------------------------
# Original file get_train.py from the Kaggle repo:       
import os
import sys
import subprocess

# From the origincal file:
if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

"""
convert is an ImageMagick function, but I couldn't get it to work 
Looking at http://www.mechanicalgirl.com/view/image-resizing-python-and-imagemagick-bleh/
"""
"""
This seems to be the issue:
http://stackoverflow.com/questions/15016974/running-imagemagick-convert-console-application-from-python
I figured it out: It turns out that windows has its own convert.exe program in
PATH.
C:\Users\Navin>where convert                                                    
C:\Program Files\ImageMagick-6.8.3-Q16\convert.exe                              
C:\Windows\System32\convert.exe     

convert = subprocess.check_call(["convert", vars['source_image_fullpath'], \
    "-scale", vars['sizes'][size]['width'] + 'x' + vars['sizes'][size]['height'] + '!',  new_path])
convert_results = str(convert)
 
convert = subprocess.check_call(["convert 64.jpg -scale 48x48! 64_48x48.jpg"])
"""
#cmd = "convert -resize 48x48\\! "
cmd = "convert -resize 48x48\! "
#cmd = "convert -resize 48x48 "
# insert the Pillow or PIL command to resize
classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
#for cls in classes[2:3]:
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(fi + cls)        
    for img in imgs:
#    for img in imgs[1:2]:
        md = ""
        md += cmd    
        md += fi + cls + "\\" + img
        md += " " + fo + cls + "\\" + img
        os.system(md)

