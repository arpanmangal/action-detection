"""
Script to implement TC proposed by COIN
Usage:
$ python task_consistency.py <Mode> <W_MATRIX_PATH> <IN_PKL> <OUT_PKL> <TAG_FILE>
"""

import pickle
import sys
import numpy as np
import math

assert len(sys.argv) == 6

mode = int(sys.argv[1])
W = np.load(sys.argv[2])
in_pkl = sys.argv[3]
out_pkl = sys.argv[4]
tag_file = sys.argv[5]

if mode == 0:
    print ("Mode: W")
elif mode == 1:
    print ("Mode: MTL")
elif mode == 2:
    print ("Mode: GLCU-TH")
elif mode == 3:
    print ("Mode: GLCU-G")
else:
    raise ValueError("Invalid mode value")

ssn_scores = pickle.load(open(in_pkl, 'rb'))
pruned_scores = dict()
task_predictions = dict()

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]

def prune_scores(scores, vid_id):
    props, act_scores, comp_scores, regs = scores[:4]
    N, K = comp_scores.shape
    combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
    T = W.shape[1]
    if mode == 0:
        step_scores = combined_scores.mean(axis=0).reshape(1, -1)
        assert W.shape[0] == K
        task = np.argmax(np.matmul(step_scores, W))
    elif mode < 3:
        assert len(scores[-1]) == T
        task = np.argmax(scores[-1])
    else:
        assert len(scores[-2]) == T
        task = np.argmax(scores[-2])

    # Mask step scores
    mask = np.full(combined_scores.shape, math.exp(-2))
    mask[:,0] = 1
    mask[:, np.where(W.T[task])[0]] = 1
    combined_scores *= mask

    return scores[:1] + (combined_scores, None) + scores[3:], task

def get_task_gt(tag_file):
    """
    Retrieve task ground truths for task classification accuracy computation
    """
    task_gt = dict()
    f = open(tag_file, 'r')
    while (len(f.readline()) > 0):
        vid_id = f.readline().strip().split('/')[-1]
        f.readline()
        task_id = int(f.readline().strip())
        f.readline()

        corrects = int(f.readline().strip())
        for _c in range(corrects):
            f.readline()

        preds = int(f.readline().strip())
        for _p in range(preds):
            f.readline()

        task_gt[vid_id] = task_id
    
    return task_gt

def task_acc(scores, preds):
    """
    Find the task classification accuracy
    """
    gt = get_task_gt(tag_file)
    combined = np.array([[task, preds[vid_id]] for vid_id, task in gt.items()], dtype=int)
    acc = sum(combined[:, 0] == combined[:, 1]) / combined.shape[0]
    print ('Prediction Accuracy: %.4f' % (acc * 100))

def task_glcu_acc(scores, index=-1):
    """
    Find the task classification accuracy for GLCU
    """
    gt = get_task_gt(tag_file)
    pred = {vid_id:np.argmax(vid_scores[index]) for vid_id, vid_scores in scores.items()}

    combined = np.array([[task, pred[vid_id]] for vid_id,task in gt.items()], dtype=int)
    acc = sum(combined[:, 0] == combined[:, 1]) / combined.shape[0]
    return acc * 100
    print ('GLCU Task Accuracy: %.4f' % (acc * 100))

for vid_id, vid_scores in ssn_scores.items():
    vid_id = vid_id.split('/')[-1]
    pruned_scores[vid_id], task_predictions[vid_id] = prune_scores(vid_scores, vid_id)

pickle.dump(pruned_scores, open(out_pkl, 'wb'), protocol=-1)
task_acc(pruned_scores, task_predictions)

if mode > 0:
    print ('MTL Task Accuracy:', task_glcu_acc(pruned_scores, -1))
if mode > 1:
    print ('GLCU Task Accuracy:', task_glcu_acc(pruned_scores, -2))
    