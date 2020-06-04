"""
Combines various result pickle files with a subset of the test results
Need: ssn_test_bp takes long time to run, so test TAG files broken into multiple pieces and tested.
    This script combines the results
Usage:
$ python combine_pkl_files.py <RESULT_DIR> <NUM_SPLIT>
"""

import os
import sys
import pickle

assert len(sys.argv) >= 3
result_dir = sys.argv[1]
num_split = int(sys.argv[2])
if len(sys.argv) > 3:
    start = int(sys.argv[3])
else:
    start = 0

# Combining the result file
result = dict()
for i in range(start, num_split+start):
    result.update(pickle.load(open(os.path.join(result_dir, 'result%d.pkl' % i),'rb')))
pickle.dump(result, open(os.path.join(result_dir, 'result.pkl'), 'wb') )

# Combining the bo file
## Not need right now...