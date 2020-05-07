"""
Script to transfer trained-weights from small sub-models to main model
"""


import argparse
import torch

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Transfer weights of various sub-models")

    parser.add_argument('base', type=str, help='Path of the base model')
    parser.add_argument('submodel', type=str, help='Path of the sub-model')
    parser.add_argument('savepth', type=str, help='Checkpoint path where to save the model')
    parser.add_argument('--task_head', default=False, action='store_true', help='Transfer Task Head (GLCU) weights')
    parser.add_argument('--cls_head', default=False, action='store_true', help='Transfer Cls Head weights')
    parser.add_argument('--mid_glcu', default=False, action='store_true', help='Transfer middle-GLCU weights')
    parser.add_argument('--reverse', default=False, action='store_true', help='Reverse mode - transfer weights from base to submodel')

    args = parser.parse_args()
    return args


def load_model(checkpoint_pth):
    """Read the task and ssn weights"""
    if torch.cuda.is_available():
        model = torch.load(checkpoint_pth)
    else:
        model = torch.load(checkpoint_pth, map_location=torch.device('cpu'))
    return model


def get_top_key(key):
    return key.split('.')[0]


def get_top_keys(state_dict):
    """Returns the top keys in the state_dict"""
    return set({get_top_key(k) for k in state_dict.keys()})


def transfer_weights(base, submodel, save_checkpoint_pth, task_head=False, cls_head=False, mid_glcu=False, reverse=False):
    """
    Transfer weights from submodel to base
    """
    if cls_head:
        raise NotImplementedError

    assert task_head or mid_glcu # At-least one is true
    assert not (task_head and mid_glcu) # Not both should be true

    base_model = load_model(base)
    submodel_weights = load_model(submodel)
    
    if task_head:
        for key, weight in submodel_weights.items():
            if task_head: base_key = 'module.task_head.' + key
            if mid_glcu: base_key = 'module.glcu.' + key
            if reverse:
                submodel_weights[key] = base_model['state_dict'][base_key]
            else:
                base_model['state_dict'][base_key] = weight

    if reverse:
        torch.save(submodel_weights, save_checkpoint_pth)
    else:
        torch.save(base_model, save_checkpoint_pth)


if __name__ == '__main__':
    args = parse_args()
    transfer_weights(args.base, args.submodel, args.savepth, task_head=args.task_head, mid_glcu=args.mid_glcu, reverse=args.reverse)