# -*- coding: utf-8 -*-
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will
#   list the files in the input directory from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


#ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with
#  many modifications to enhance the performance. First it use FAST to find
#  keypoints, then apply Harris corner measure to find top N points among them.
#For any feature set of n binary tests at location (x_i, y_i),
#  define a 2 \times n matrix, S which contains the coordinates of these pixels.
#    Then using the orientation of patch, \theta, its rotation matrix is found
#      and rotates the S to get steered(rotated) version S_\theta.
#ORB runs a greedy search among all possible binary tests to find the ones that
# have both high variance and means close to 0.5, as well as being uncorrelated.

# Any results write to the current directory are saved as output.

import numpy as np  # linear algebra
import cv2
import os

import csv
import sys
from time import sleep


def im_align_orb(imp1, imp2, nf=10000):
    """
    :param imp1: image1 file path
    :param imp2: image2 file path
    :param nf: max number of ORB key points
    :return:  transformed image2, so that it can be aligned with image1
    """
    img1 = cv2.imread(imp1, 0)
    img2 = cv2.imread(imp2, 0)
    h2, w2 = img2.shape[:2]

    orb = cv2.ORB_create(nfeatures=nf, WTA_K=2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des1, des2, 2)

    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))

    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1, 1, 2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(kp2_, kp1_, cv2.RANSAC, 1.0)

    h1, w1 = img1.shape[:2]

    img2 = cv2.warpPerspective(cv2.imread(imp2), H, (w1, h1))
    return img2


def align_set_by_id(setid, setvalue, isTrain=True, nFeatures=20000):
    """
    :param setid: image set id values
    :param isTrain: train (true) or test (false) path
    :return: aligned images into output path
    """
    train_path = '../output/train_sm/'
    test_path = '../output/test_sm/'

    counter = 0

    if isTrain:
        image_path = train_path
        fn1 = train_path + "set" + key + "_" + elem[0] + ".jpg"
        outputpath = "./train_output/"
    else:
        image_path = test_path
        fn1 = train_path + "set" + key + "_" + elem[0] + ".jpg"
        print(fn1)
        outputpath = "./test_output/"

    result = list()

    result.append(cv2.cvtColor(cv2.imread(fn1), cv2.COLOR_BGR2RGB))
    for id in elem:  # outputmatrix elem
        fn2 = image_path + "set" + str(setid) + "_" + str(id) + ".jpg"
        print("fn1=%s, fn2=%s" % (os.path.basename(fn1), os.path.basename(fn2)))
        im = im_align_orb(fn1, fn2, nFeatures)
        cv2.imwrite(outputpath + os.path.basename(fn2), im)
        result.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        counter += 1
        for i in range(21):
            sys.stdout.write('\r')
            sys.stdout.write(
                '[%-20s] %d%% %d/%d ' % ('=' * i, 5 * i, counter, om_len)
                )
            sys.stdout.flush()
            sleep(0.25)

    return result


def align_all_set(path, isTrain=True):
    allfiles = os.listdir(path)
    allfiles = [
        os.path.basename(file) for file in allfiles if file.startswith('set')]
    allsets = np.unique([f.split("_")[0].replace("set", "") for f in allfiles])

    for s in allsets:
        align_set_by_id(s, isTrain=True, nFeatures=20000)

#align_all_set(path='../output/train_sm')


def csv_lists(path):
    row = []
    matrix = {}

    with open(path) as f:
        csv_reader = csv.reader(f)
        csv_list = list(csv_reader)

    for idx, val in enumerate(csv_list):
        if not row:
            row.extend([val[0]])
        if row[0] == val[0]:
            row.extend([val[1]])
        elif row != val[0]:
            row = [val[0]]
            row.extend([val[1]])
        if len(row) is 6:
            matrix.update({row[0]: row[1:]})
    return matrix

outputmatrix = csv_lists('../output/features_means_train.csv')
om_len = len(outputmatrix)

for key, elem in list(outputmatrix.items()):
    align_set_by_id(key, elem, isTrain=True, nFeatures=15000)