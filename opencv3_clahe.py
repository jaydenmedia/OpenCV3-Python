# -*- coding: utf-8 -*-
"""Example opencv3_clahe.py

This module allows users to apply CLAHE to images using OpenCV and Python.


Read more about CLAHE:
    http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html


Examples:
    There are two modes, clahe() and rgb_clahe().  Images may be processed
        individually or as part of a batch process.


    Default applies CLAHE to the overall image.

    Single image, grayscale::
        $ python clahe("image.tif")


    RGB mode applies CLAHE to the RGB channels individually.

    Single image, rgb::
        #python rgb_clahe("image.tif")


    Batch mode defaults to grayscale unless the optional rgb parameter
        is selected.

    Batch mode, with optional rgb parameter selected::
        #python batch_clahe("path/to/images/directory", rgb)


The output directory, file name, and extension are created by changing the
    values of output_name and output_img_ext.


In limited testing, successful images extensions are .tif and .jpg.


Notes:
    OpenCV converts RGB as BGR by default.
    Images will return "ValueError: need more than 0 values to unpack"
        if the uint value is not in the proper bit format or color space.



Attributes:
    clahe (string): Applies CLAHE to an image and outputs the result in a
        grayscale format.

    rgb_clahe (string): Applies CLAHE to an image and outputs the result in an
        rgb format.

Todo:
    * Test images with extensions other than .tif and .jpg
    * Set a mechanism to ensure images are within the proper context when
        performing cv2.split() in order to avoid a ValueError: 0 values.


.. codeauthor:: Josh Smith <jaydenmedia.com>

"""

import os
import cv2

import sys
from time import sleep

output_prefix = "clahe"
output_img_ext = ".jpg"
output_folder = "opencv3_clahe_output"


def output_dir():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def clahe(img):
    image = cv2.imread(img, 0)
    img_name = os.path.splitext(img)[0]
    newimage = output_prefix + img_name + output_img_ext

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)

    output_dir()
    cv2.imwrite(os.path.join(output_folder, newimage), cl1)


def rgb_clahe(img):
    bgr_image = cv2.imread(img)
    img_name = os.path.splitext(img)[0]
    newimage = output_prefix + "-rgb-" + img_name + output_img_ext

    b_channel, g_channel, r_channel = cv2.split(bgr_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cl1 = clahe.apply(b_channel)
    cl2 = clahe.apply(g_channel)
    cl3 = clahe.apply(r_channel)
    cv2.merge([cl1, cl2, cl3], bgr_image)

    output_dir()
    cv2.imwrite(os.path.join(output_folder, newimage), bgr_image)


def batch_clahe(imgdir, mode=None):
    counter = 0
    pick_path = imgdir
    allfiles = os.listdir(pick_path)
    n_files = len(allfiles)

    if mode is "rgb":
        for filename in os.listdir(imgdir):
            print(filename)
            rgb_clahe(filename)

            counter += 1
            for i in range(21):
                sys.stdout.write('\r')
                sys.stdout.write(
                    '[%-20s] %d%% %d/%d %s   ' % (
                        '=' * i, 5 * i, counter, n_files, filename)
                    )
                sys.stdout.flush()
                sleep(0.25)

    else:
        for filename in os.listdir(imgdir):
            print(filename)
            clahe(filename)

            counter += 1
            for i in range(21):
                sys.stdout.write('\r')
                sys.stdout.write(
                    '[%-20s] %d%% %d/%d %s   ' % (
                        '=' * i, 5 * i, counter, n_files, filename)
                    )
                sys.stdout.flush()
                sleep(0.25)


if __name__ == '__main__':

# Uncomment selected command and then run file.

#   Grayscale example
#    clahe("green-python.jpg")

#   RGB example
#    rgb_clahe("green-python.jpg")

#   Batch example with optional RGB parameter selected
    batch_clahe("original/images/folder/path", "rgb")
