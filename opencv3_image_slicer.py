# -*- coding: utf-8 -*-
import image_slicer
import os
import sys
from time import sleep

train_path = 'train_output/'
test_path = 'test_output/'
pick_path = train_path  # select path to sort

dest_train = 'train_matrices'
dest_test = 'test_matrices'
pick_dest = dest_train  # select destination path

allfiles = os.listdir(pick_path)
file_count = len(allfiles)
counter = 0

for f in allfiles:
    tiles = image_slicer.slice(pick_path + f, 20, save=False)
    image_slicer.save_tiles(tiles, directory=pick_dest,
                            prefix=f)
    counter += 1
    for i in range(21):
        sys.stdout.write('\r')
        sys.stdout.write(
            '[%-20s] %d%% %d/%d %s   ' % (
                '=' * i, 5 * i, counter, file_count, f)
            )
        sys.stdout.flush()
        sleep(0.25)

# image_slicer.save_tiles(tiles, directory='~/cake_slices',\
#                            prefix='slice', ext='jpg')