"""
Script for performing TAG pruning
Usage:
$ python tag_pruning.py <IN_PKL> <OUT_PKL>
"""

import pickle
import sys

assert len(sys.argv) >= 3

in_pkl = sys.argv[1]
out_pkl = sys.argv[2]
lo_thres = 0.05 if len(sys.argv) <= 4 else float(sys.argv[3])
hi_thres = 0.6 if len(sys.argv) <= 4 else float(sys.argv[4])
assert 0.0 <= lo_thres < hi_thres < 1.0

ssn_scores = pickle.load(open(in_pkl, 'rb'))
pruned_scores = dict()

def prune_scores(scores, vid_id, task=False):
    if task:
        raise NotImplementedError

    (props, act_scores, comp_scores, regs) = scores
    keep = [idx for idx, p in enumerate(props) if (lo_thres < p[1] - p[0] < hi_thres)]
    if len(keep) == 0:
        print ('video %s is completely useless!' % vid_id)
        return scores # Keep all the elements

    return props[keep, :], act_scores[keep, :], comp_scores[keep, :], regs[keep, :, :]


for vid_id, vid_scores in ssn_scores.items():
    vid_id = vid_id.split('/')[-1]
    pruned_scores[vid_id] = prune_scores(vid_scores, vid_id)

pickle.dump(pruned_scores, open(out_pkl, 'wb'), protocol=-1)