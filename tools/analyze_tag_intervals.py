"""
Script to analyze the ground-truth interval distribution
Usage:
$ python analyze_tag_intervals.py <IN_TAG_FILE>
"""

import sys
import numpy as np

assert len(sys.argv) == 2

in_tag = sys.argv[1]

def read_block (tag_file):
    # Read TAG blocks
    print ('Processing %s TAG file' % tag_file)
    f = open(tag_file, 'r')

    while (len(f.readline()) > 0):
        # Keep reading the block
        f.readline() # ID
        n_frames = int(f.readline().strip()) # Num Frames
        f.readline() # Task ID
        f.readline() # Useless

        cintervals = []
        corrects = int(f.readline().strip()) # Num correct
        for _c in range(corrects):
            annotation = f.readline().strip().split(' ')
            i0 = float(annotation[1]) / n_frames
            i1 = float(annotation[2]) / n_frames
            cintervals.append(float('%.3f' % (i1 - i0)))
        
        pintervals = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            annotation = f.readline().strip().split(' ')
            i0 = float(annotation[3]) / n_frames
            i1 = float(annotation[4]) / n_frames
            pintervals.append(float('%.3f' % (i1 - i0)))
        yield cintervals, pintervals


gt_intervals = []
tag_intervals = []
for cint, pint in read_block(in_tag):
    gt_intervals += cint
    tag_intervals += pint

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


print (len(gt_intervals))
print (len(tag_intervals))

# print (gt_intervals[:100])
# print (tag_intervals[:100])

bins = np.linspace(0, 1.0, 21)
print (bins)
gt_dist = np.histogram(gt_intervals, bins)[0]
print (gt_dist)
print (np.cumsum(gt_dist / np.sum(gt_dist)))
tag_dist = np.histogram(tag_intervals, bins)[0]
print (tag_dist)
print (np.cumsum(tag_dist / np.sum(tag_dist)))
