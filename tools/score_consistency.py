"""
Script to score consistency of a given model between the steps and tasks
Usage:
$ python score_consistency.py <W_MATRIX_PATH> <PKL_RESULT_FILE> [--TC_MODE] [--task_mode]

TC_MODE (0/1): Indicates whether TC is already performed (default: 0)
task_mode (0/1): Indicates whether task scores are available or are to be computed by TC (default: 1)
"""

import pickle
import sys
import numpy as np

class AverageMeter:
    """Computes and stores the average value"""
    def __init__(self, size):
        if size == 1:
            self.avg = 0.0
        else:
            self.avg = np.zeros(size, dtype=float)
        self.count = 0

    def update(self, val):
        self.avg = (self.avg * self.count + val) / (self.count + 1)
        self.count += 1


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]

# def get_pr_scores(gt, pred):
#     """
#     Computes the confusion matrix:
#     [[TP FN]
#      [FP TN]]
#     returning as [TP FN FP TN]
#     gt & pred are vectors of class present / absent
#     """
#     assert gt.shape == pred.shape and gt.shape[0] > 1
#     gt = gt.astype(bool)
#     pred = pred.astype(bool)
#     ngt = np.logical_not(gt)
#     npred = np.logical_not(pred)

#     TP = np.sum(pred[gt])
#     FN = np.sum(npred[gt])
#     FP = np.sum(pred[ngt])
#     TN = np.sum(npred[ngt])
#     total_pos = TP + FN
#     total_neg = FP + TN
#     assert total_pos + total_neg == gt.shape[0]

#     return np.array([TP/total_pos, FN/total_pos, FP/total_neg, TN/total_neg], dtype=float)

def get_acc_scores(allowed, pred):
    """
    computes the correct prediction accuracy = [TP / (TP + FP)]
    """
    assert allowed.shape == pred.shape and allowed.shape[0] > 1
    allowed = allowed.astype(bool)
    nallowed = np.logical_not(allowed)

    correct = np.sum(pred[allowed])
    wrong = np.sum(pred[nallowed])

    return correct / (correct + wrong)


if __name__ == '__main__':
    assert len(sys.argv) == 3 or len(sys.argv) == 5

    W = np.load(sys.argv[1])
    ssn_scores = pickle.load(open(sys.argv[2], 'rb'))
    tc_mode = 0
    task_mode = 1
    if len(sys.argv) > 3:
        tc_mode = int(sys.argv[3])
        task_mode = int(sys.argv[4])
    assert 0 <= tc_mode <= 1 and 0 <= task_mode <= 1

    # avg_pr_scores = AverageMeter(4)
    avg_acc = AverageMeter(1)

    for k, scores in ssn_scores.items():
        if tc_mode == 1:
            combined_scores = scores[1]
        else:   
            props, act_scores, comp_scores, regs = scores[:4]
            combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
        N, K = combined_scores.shape
        step_predictions = np.argmax(combined_scores, axis=1)

        assert W.shape[0] == K
        T = W.shape[1]
        if task_mode == 0:
            step_scores = combined_scores.mean(axis=0).reshape(1, -1)
            task = np.argmax(np.matmul(step_scores, W))
        else:
            assert len(scores[-1]) == T
            task = np.argmax(scores[-1].squeeze())

        # gt = W[:, task]
        # pred = np.zeros(K, dtype=int)
        # for p in step_predictions:
        #     pred[p] = 1
        allowed = W[:, task]
        pred = np.zeros(K, dtype=int)
        for p in step_predictions:
            pred[p] += 1

        assert allowed.shape == pred.shape
        
        # pr_score = get_pr_scores(gt, pred)
        # avg_pr_scores.update(pr_score)
        acc_score = get_acc_scores(allowed, pred)
        avg_acc.update(acc_score)

        
    # print ('#Videos:', avg_pr_scores.count)
    # print ('Confusion Matrix:')
    # c = avg_pr_scores.avg * 100
    # print ('%.2f %.2f\n%.2f %.2f' % tuple(c))
    print ('#Videos:', avg_acc.count)
    print ('Avg. Acc. %.2f' % (avg_acc.avg * 100))