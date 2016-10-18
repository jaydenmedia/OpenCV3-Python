# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
from time import sleep

MIN_MATCH_COUNT = 10
#################


def log_sets():
    target = 'train_matrices/'
    allfiles = os.listdir(target)
    sift = cv2.xfeatures2d.SIFT_create()
    outfile = "train_set_matrix_features_ranking.txt"

    os.chdir(target)
    n_files = len([name for name in os.listdir('.') if os.path.isfile(name)])
    counter = 0

    for s in allfiles:
        img = cv2.imread(s)          # queryImage
        (kps, descs) = sift.detectAndCompute(img, None)
        with open(outfile, 'a+') as f:
            f.write(s)
            f.write(',')
            f.write('# kps: {}, descriptors: {}'.format(len(kps), descs.shape))
            f.write('\n')
        counter += 1
        for i in range(21):
            sys.stdout.write('\r')
            sys.stdout.write(
                '[%-20s] %d%% %d/%d ' % ('=' * i, 5 * i, counter, n_files)
                )
            sys.stdout.flush()
            sleep(0.25)
    sys.stdout.write('Done!')

log_sets()
