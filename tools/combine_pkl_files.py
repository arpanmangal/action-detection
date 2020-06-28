"""
Combines various result pickle files with a subset of the test results
Need: ssn_test_bp takes long time to run, so test TAG files broken into multiple pieces and tested.
    This script combines the results
Usage:
$ python combine_pkl_files.py <RESULT_DIR> <SPLIT_START> <SPLIT_END> <FB_START> <FB_END>
"""

import os
import sys
import pickle

assert len(sys.argv) >= 4
result_dir = sys.argv[1]
split_start = int(sys.argv[2])
split_end = int(sys.argv[3])
fb = False
if len(sys.argv) >= 6:
    fb = True
    fb_start = int(sys.argv[4])
    fb_end = int(sys.argv[5])

# Combining the result file
if not fb:
    result = dict()
    for i in range(split_start, split_end+1):
        result.update(pickle.load(open(os.path.join(result_dir, 'result%d.pkl' % i),'rb')))
    pickle.dump(result, open(os.path.join(result_dir, 'result.pkl'), 'wb') )
else:
    for fb in range(fb_start, fb_end+1):
        result = dict()
        for i in range(split_start, split_end+1):
            result.update(pickle.load(open(os.path.join(result_dir, 'result%d_fb%d.pkl' % (i, fb)),'rb')))
        pickle.dump(result, open(os.path.join(result_dir, 'result_fb%d.pkl' % fb), 'wb') )

# Combining the bo file
if not fb:
    result_f = open(os.path.join(result_dir, 'result_bo.pkl'), 'wb')
    for i in range(split_start, split_end+1):
        f = open(os.path.join(result_dir, 'result_bo%d.pkl' % i), 'rb')
        while True:
            try:
                vid_id, base_out = pickle.load(f)
                pickle.dump((vid_id, base_out), result_f)
            except EOFError:
                break
            