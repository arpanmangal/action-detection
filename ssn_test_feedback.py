import argparse
import time

import numpy as np
import pickle
import sys

from ssn_dataset import SSNDataSet
from ssn_models import SSN
from transforms import *
from ops.ssn_ops import STPPReorgainzed
import torch
from torch import multiprocessing, optim
from torch.utils import model_zoo
from ops.utils import get_configs, get_reference_model_url
import torch.nn.functional as F


parser = argparse.ArgumentParser(
    description="SSN Testing Tool")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'thumos14', 'coin'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)
parser.add_argument('--feedback', type=int, default=5, help='Number of feedbacks')
parser.add_argument('--direct', action='store_true', default=False, help='Whether to directly use the one-hot final task prediction')
parser.add_argument('--test_prop_file', type=str, default=None, help='Path of test TAG file. If None will be taken from configs')
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--data_root', type=str, default='data/rawframes',
                    metavar='PATH', help='path of the rawframes folder')
parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--save_base_out', type=str, default=None)
parser.add_argument('--aug_ratio', type=float, default=0.5)
parser.add_argument('--frame_interval', type=int, default=6)
parser.add_argument('--test_batchsize', type=int, default=512)
parser.add_argument('--no_regression', action="store_true", default=False)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_pref', type=str, default='')
parser.add_argument('--use_reference', default=False, action='store_true')
parser.add_argument('--use_kinetics_reference', default=False, action='store_true')

args = parser.parse_args()

dataset_configs = get_configs(args.dataset)

num_class = dataset_configs['num_class']
num_tasks = dataset_configs['num_tasks']
stpp_configs = tuple(dataset_configs['stpp'])
if args.test_prop_file is None:
    args.test_prop_file = dataset_configs['test_list']
test_prop_file = 'data/{}_proposal_list.txt'.format(args.test_prop_file)

if args.modality == 'RGB':
    data_length = 1
elif args.modality in ['Flow', 'RGBDiff']:
    data_length = 5
else:
    raise ValueError("unknown modality {}".format(args.modality))

gpu_list = args.gpus if args.gpus is not None else range(8)


def runner_func(dataset, state_dict, stats, gpu_id, index_queue, result_queue):
    torch.cuda.set_device(gpu_id)

    while True:
        index = index_queue.get()

        net = SSN(num_class, num_tasks, 2, 5, 2,
              args.modality, test_mode=True,
              base_model=args.arch, no_regression=args.no_regression, stpp_cfg=stpp_configs,
              task_head=True, glcu_skip=True, verbose=False)
        net.load_state_dict(state_dict)
        net.prepare_test_fc_skip_glcu()
        net.cuda()
        output_dim = net.test_fc.out_features
        base_out_dim = net.base_feature_dim
        reorg_stpp = STPPReorgainzed(output_dim, num_class + 1, num_class,
                                 num_class * 2, True, stpp_cfg=stpp_configs)

        # Do simple inference to get backbone predictions
        inf_dataset = dataset
        net.eval()
        frames_gen, frame_cnt, rel_props, prop_ticks, prop_scaling = inf_dataset[index]

        def backbone():
            """
            Do simple inference to get backbone predictions
            """
            num_crop = args.test_crops
            length = 3
            if args.modality == 'Flow':
                length = 10
            elif args.modality == 'RGBDiff':
                length = 18
    
            # First get the base_out outputs
            base_output = torch.autograd.Variable(torch.zeros((num_crop, frame_cnt, base_out_dim)).cuda(),
                                                    volatile=True)
            cnt = 0
            for frames in frames_gen:
                # frames.shape == [frame_batch_size * num_crops * 3, 224, 224]
                # frame_batch_size is 4 by default
                input_var = torch.autograd.Variable(frames.view(-1, length, frames.size(-2), frames.size(-1)).cuda(),
                                                    volatile=True)
                base_out = net(input_var, None, None, None, None)
                bsc = base_out.view(num_crop, -1, base_out_dim)
                base_output[:, cnt:cnt+bsc.size(1), :] = bsc
                cnt += bsc.size(1)

            n_frames = base_output.size(1)
            assert frame_cnt == n_frames
            # GLCU
            step_features = base_output.mean(dim=0).mean(dim=0).unsqueeze(0)
            glcu_task_pred = net.glcu_asc(step_features)
            assert glcu_task_pred.size(0) == 1
            glcu_task_pred = F.softmax(glcu_task_pred, dim=1)

            return base_output, glcu_task_pred

        def feedback(backbone_output, task_pred_glcu):
            """
            Run the classifier and STPP using backbone and glcu output
            """
            # output.shape == [num_frames, 7791]
            output = torch.zeros((frame_cnt, output_dim)).cuda()
            cnt = 0
            num_crop = args.test_crops
            for i in range(0, frame_cnt, 4):
                base_out = backbone_output[:, i:i+4, :].contiguous().view(-1, base_out_dim)
                num_frames = base_out.size(0)
                glcu_task_pred_4_times = task_pred_glcu.repeat(1, num_frames).view(num_frames, task_pred_glcu.size(1))
                base_out = torch.cat((base_out, glcu_task_pred_4_times), dim=1)

                rst = net.test_fc(base_out)
                sc = rst.data.view(num_crop, -1, output_dim).mean(dim=0)
                output[cnt: cnt + sc.size(0), :] = sc
                cnt += sc.size(0)

            # act_scores.shape == [num_proposals, K+1]
            # comp_scores.shape == [num_proposals, K]
            act_scores, comp_scores, reg_scores = reorg_stpp.forward(output, prop_ticks, prop_scaling)
            act_scores = torch.autograd.Variable(act_scores, volatile=True)
            comp_scores = torch.autograd.Variable(comp_scores, volatile=True)

            # Task Head
            combined_scores = F.softmax(act_scores[:, 1:], dim=1) * torch.exp(comp_scores)
            combined_scores = combined_scores.mean(dim=0).unsqueeze(0)
            task_pred = F.softmax(net.task_head(combined_scores).squeeze(), dim=0).data.cpu().numpy()

            act_scores = act_scores.data
            comp_scores = comp_scores.data

            if reg_scores is not None:
                reg_scores = reg_scores.view(-1, num_class, 2)
                reg_scores[:, :, 0] = reg_scores[:, :, 0] * stats[1, 0] + stats[0, 0]
                reg_scores[:, :, 1] = reg_scores[:, :, 1] * stats[1, 1] + stats[0, 1]

            torch.cuda.empty_cache() # To empty the cache from previous iterations

            return act_scores, comp_scores, reg_scores, task_pred

        # Step 1: Run the backbone inference
        base_output, glcu_task_pred = backbone()
        torch.cuda.empty_cache() # To empty the cache from previous iterations

        # Step 2: Run the feedback
        feedbacks = []
        act_scores, comp_scores, reg_scores, task_pred = feedback(base_output, glcu_task_pred)
        feedbacks.append((rel_props.cpu().numpy(), act_scores.cpu().numpy(), comp_scores.cpu().numpy(), reg_scores.cpu().numpy(), 
                           glcu_task_pred.squeeze().data.cpu().numpy(), task_pred))

        for fb in range(args.feedback):
            # Use the last task_pred as GLCU_desc output
            if args.direct:
                task_pred_glcu = torch.autograd.Variable(torch.zeros(1, num_tasks)).cuda()
                y = int(task_pred.argmax(axis=0))
                task_pred_glcu[0, y] = 1.0
            else:
                task_pred_glcu = torch.autograd.Variable(torch.from_numpy(task_pred).view(1, num_tasks)).cuda()
            
            act_scores, comp_scores, reg_scores, task_pred = feedback(base_output, task_pred_glcu)
            feedbacks.append((rel_props.cpu().numpy(), act_scores.cpu().numpy(), comp_scores.cpu().numpy(), reg_scores.cpu().numpy(), 
                           task_pred_glcu.squeeze().data.cpu().numpy(), task_pred))

        base_output = base_output.mean(dim=0).data

        # perform stpp on scores
        result_queue.put((inf_dataset.video_list[index].id,
            feedbacks,
            base_output.cpu().numpy()
        ))


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')  # this is crucial to using multiprocessing processes with PyTorch

    # This net is used to provides setup settings. It is not used for testing.
    net = SSN(num_class, num_tasks, 2, 5, 2,
              args.modality, test_mode=True,
              base_model=args.arch, no_regression=args.no_regression, stpp_cfg=stpp_configs,
              task_head=True, glcu_skip=True, verbose=False)

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    # Use single crop for KL loss backprop for 10X faster optimization
    single_cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])

    if not args.use_reference and not args.use_kinetics_reference:
        checkpoint = torch.load(args.weights)
    else:
        raise ValueError("Sorry, please use only the trained models!")
        model_url = get_reference_model_url(args.dataset, args.modality,
                                            'ImageNet' if args.use_reference else 'Kinetics', args.arch)
        checkpoint = model_zoo.load_url(model_url)
        print("using reference model: {}".format(model_url))

    print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['best_loss']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    stats = checkpoint['reg_stats'].numpy()

    dataset = SSNDataSet(args.data_root, test_prop_file,
                         new_length=data_length,
                         modality=args.modality,
                         aug_seg=2, body_seg=5,
                         image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB",
                                                                          "RGBDiff"] else args.flow_pref + "{}_{:05d}.jpg",
                         test_mode=True, test_interval=args.frame_interval,
                         transform=torchvision.transforms.Compose([
                             cropping,
                             Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                             ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                             GroupNormalize(net.input_mean, net.input_std),
                         ]), reg_stats=stats, verbose=False)

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func, args=(dataset, base_dict, stats, gpu_list[i % len(gpu_list)], index_queue, result_queue))
               for i in range(args.workers)]

    for w in workers:
        w.daemon = True
        w.start()

    max_num = args.max_num if args.max_num > 0 else len(dataset)

    for i in range(max_num):
        index_queue.put(i)

    proc_start_time = time.time()
    if args.save_base_out is not None: base_out_f = open(args.save_base_out, 'wb')
    # if args.save_raw_scores is not None: raw_score_f = open(args.save_raw_scores, 'wb')
    out_dict = dict()
    for fb in range(args.feedback + 1):
        out_dict[fb] = dict()

    for i in range(max_num):
        vid_id, feedbacks, base_out = result_queue.get()
        vid_id = vid_id.split('/')[-1]
        if args.save_base_out is not None: pickle.dump((vid_id, base_out), base_out_f, pickle.HIGHEST_PROTOCOL)

        for fb, rst in enumerate(feedbacks):
            out_dict[fb][vid_id] = rst

        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {:.04f} sec/video'.format(i, i + 1,
                                                                        max_num,
                                                                        float(cnt_time) / (i + 1)))
    if args.save_base_out is not None: base_out_f.close()
    # if args.save_raw_scores is not None: raw_score_f.close()

    if args.save_scores is not None:
        for fb in range(args.feedback + 1):
            save_file = '%s_fb%d.pkl' % (args.save_scores, fb)
            save_dict = out_dict[fb]
            pickle.dump(save_dict, open(save_file, 'wb'), pickle.HIGHEST_PROTOCOL)
