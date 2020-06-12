"""
Script to set the weights of descending GLCU artificially
Usage: python artificial_desc_glcu.py <PATH_W_MATRIX> <PATH_BASE_MODEL> <PATH_SAVE>
"""

import sys
import os
import numpy as np
import torch

assert len(sys.argv) == 4
W_path = sys.argv[1]
base_path = sys.argv[2]
save_path = sys.argv[3]

# Load the matrix
W = np.load(W_path).astype(int)

# Load the model
if torch.cuda.is_available():
    model = torch.load(base_path)
else:
    model = torch.load(base_path, map_location=torch.device('cpu'))

for k in ['module.glcu_dsc_act.fcs.0.weight', 'module.glcu_dsc_comp.fcs.0.weight', 'module.glcu_dsc_reg.fcs.0.weight']:
    assert k in model['state_dict'].keys()

# Transfer completeness weights
old_comp = model['state_dict']['module.glcu_dsc_comp.fcs.0.weight']
new_comp = torch.ones(old_comp.size()) * -1
if torch.cuda.is_available: new_comp = new_comp.cuda()
assert new_comp.size() == W.shape
for r_idx, r in enumerate(W):
    for c_idx, c in enumerate(r):
        if c == 1:
            new_comp[r_idx][c_idx] = 0
model['state_dict']['module.glcu_dsc_comp.fcs.0.weight'] = new_comp

# Transfer activity weights
old_act = model['state_dict']['module.glcu_dsc_act.fcs.0.weight']
new_act = torch.ones(old_act.size()) * -1
if torch.cuda.is_available: new_act = new_act.cuda()
assert new_act[1:, :].size() == W.shape
for r_idx, r in enumerate(W):
    for c_idx, c in enumerate(r):
        if c == 1:
            new_act[r_idx+1][c_idx] = 0
model['state_dict']['module.glcu_dsc_act.fcs.0.weight'] = new_act

torch.save(model, save_path)

