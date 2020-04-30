"""
Script to trim the video ids from a given pickle file
Usage:
$ python trim_pkl.py <IN_PKL> <OUT_PKL>
"""

import pickle
import sys

assert len(sys.argv) == 3

in_pkl = sys.argv[1]
out_pkl = sys.argv[2]

ssn_scores = pickle.load(open(in_pkl, 'rb'))
trimmed_scores = dict()

for vid_id, vid_scores in ssn_scores.items():
    trimmed_scores[vid_id.split('/')[-1]] = vid_scores

pickle.dump(trimmed_scores, open(out_pkl, 'wb'), protocol=-1)
