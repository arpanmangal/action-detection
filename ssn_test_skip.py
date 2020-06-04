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
parser.add_argument('--test_prop_file', type=str, default=None, help='Path of test TAG file. If None will be taken from configs')
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--bp', default=0, type=int, help='Number of BP steps for GLCU')
parser.add_argument('--bp_num_frames', default=220, type=int, help='Number of frames from which to predict GLCU task')
parser.add_argument('--lr', default=0.00001, type=float, help='LR for BP')
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


def runner_func(dataset, kl_train_dataset, state_dict, stats, gpu_id, index_queue, result_queue):
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

        def inference():
            """
            Do simple inference to get predictions
            """
            inf_dataset = dataset
            net.eval()
            frames_gen, frame_cnt, rel_props, prop_ticks, prop_scaling = inf_dataset[index]
            
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

            # output.shape == [num_frames, 7791]
            output = torch.zeros((frame_cnt, output_dim)).cuda()
            cnt = 0
            for i in range(0, frame_cnt, 4):
                base_out = base_output[:, i:i+4, :].contiguous().view(-1, base_out_dim)
                num_frames = base_out.size(0)
                glcu_task_pred_4_times = glcu_task_pred.repeat(1, num_frames).view(num_frames, glcu_task_pred.size(1))
                base_out = torch.cat((base_out, glcu_task_pred_4_times), dim=1)

                rst = net.test_fc(base_out)
                sc = rst.data.view(num_crop, -1, output_dim).mean(dim=0)
                output[cnt: cnt + sc.size(0), :] = sc
                cnt += sc.size(0)
            base_output = base_output.mean(dim=0).data
            glcu_task_pred = glcu_task_pred.squeeze().data.cpu().numpy()

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

            # perform stpp on scores
            return ((inf_dataset.video_list[index].id,
                    (rel_props.numpy(), act_scores.cpu().numpy(), comp_scores.cpu().numpy(), reg_scores.cpu().numpy(), 
                        glcu_task_pred, task_pred),
                    output.cpu().numpy(),
                    base_output.cpu().numpy()))

        def KL(P, Q):
            """
            find the KL divergence loss between P and Q
            """
            assert P.size() == Q.size()
            # To prevent divide by zero
            Q = Q + 1e-15
            return torch.sum(P * torch.log(P / Q))

        def update_net(optimizer):
            """
            # 1 iteration of bp
            """
            assert kl_train_dataset.bp_mode
            frames_gen, frame_cnt, rel_props, prop_ticks, prop_scaling = kl_train_dataset[index]

            optimizer.zero_grad()
            
            num_crop = 1
            length = 3
            if args.modality == 'Flow':
                length = 10
            elif args.modality == 'RGBDiff':
                length = 18
 
            for frames in frames_gen:
                # frames.shape == [frame_batch_size * num_crop * 3, 224, 224]
                assert len(frames) == length * frame_cnt
                input_var = torch.autograd.Variable(frames.view(-1, length, frames.size(-2), frames.size(-1)).cuda())
                base_out = net(input_var, None, None, None, None)
                assert base_out.size(0) == frame_cnt and base_out.size(1) == base_out_dim
                step_features = base_out.mean(dim=0).unsqueeze(0)
                glcu_task_pred = net.glcu_asc(step_features)
                glcu_task_pred_repeated = glcu_task_pred.repeat(1, frame_cnt, glcu_task_pred.size(1))
                base_out = torch.cat((base_out, glcu_task_pred_repeated), dim=1)

                output = net.test_fc(base_out)
                assert output.size(0) == frame_cnt and output.size(1) == output_dim
                act_scores, comp_scores, reg_scores = reorg_stpp.forward(output, prop_ticks, prop_scaling, bp_mode=True)

                # Task Head
                combined_scores = F.softmax(act_scores[:, 1:], dim=1) * torch.exp(comp_scores)
                combined_scores = combined_scores.mean(dim=0).unsqueeze(0)
                task_pred = net.task_head(combined_scores)
                assert task_pred.size(0) == 1
                task_pred = F.softmax(net.task_head(combined_scores).squeeze(), dim=0)

                loss = KL(task_pred, glcu_task_pred)
                loss.backward()
                torch.cuda.empty_cache() # To empty the cache from previous iterations
                break

            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            return float(loss.data), frame_cnt

        losses = []
        for bp in range(args.bp):
            # Freeze heads and create optimizer
            net.train()
            normal_weight = []
            normal_bias = []
            glcu_weight = []
            glcu_bias = []
            bn = []
            for m in net.base_model.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                    ps = list(m.parameters())
                    assert len(ps) <= 2
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    assert len(ps) <= 2
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm2d):
                    # BN layers are all frozen in SSN
                    continue
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

            for fc in net.glcu.afc:
                for m in fc.modules():
                    if isinstance(m, torch.nn.Linear):
                        ps = list(m.parameters())
                        assert len(ps) <= 2
                        glcu_weight.append(ps[0])
                        if len(ps) == 2:
                            glcu_bias.append(ps[1])
                    else:
                        print (m)
                        raise ValueError("New module type")
            
            params = [
                {'params': normal_weight, 'lr': 1 * args.lr, 'weight_decay': 1e-6,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr': 2 * args.lr, 'weight_decay': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr': 1 * args.lr, 'weight_decay': 0,
                'name': "BN scale/shift"},
                {'params': glcu_weight, 'lr': 1 * args.lr, 'weight_decay': 1e-6,
                'name': "glcu_weight"},
                {'params': glcu_bias, 'lr': 2 * args.lr, 'weight_decay': 0,
                'name': "glcu_bias"}
            ]

            optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
            optimizer.zero_grad()

            loss = update_net(optimizer)
            losses.append(loss)

        if args.bp > 0:
            print('-------------------------')
            print ('#frames: ', losses[0][1])
            for idx, loss in enumerate(losses):
                print ('%d-i%d: KL Loss' % (index, idx), loss[0])
            print('-------------------------')
            sys.stdout.flush()

        # Final inference
        net.eval()
        vid_id, (rel_props, act_scores, comp_scores, reg_scores, glcu_task_pred, task_pred), \
            output, base_output = inference()

        del net

        # perform stpp on scores
        result_queue.put((vid_id,
            (rel_props, act_scores, comp_scores, reg_scores, glcu_task_pred, task_pred),
            output, base_output   
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

    kl_train_dataset = SSNDataSet(args.data_root, test_prop_file,
                         new_length=data_length,
                         modality=args.modality,
                         aug_seg=2, body_seg=5,
                         image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB",
                                                                          "RGBDiff"] else args.flow_pref + "{}_{:05d}.jpg",
                         test_mode=True, test_interval=args.frame_interval, 
                         test_time_genbatchsize=args.bp_num_frames, bp_mode=True,
                         transform=torchvision.transforms.Compose([
                             single_cropping,
                             Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                             ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                             GroupNormalize(net.input_mean, net.input_std),
                         ]), reg_stats=stats, verbose=False)

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func, args=(dataset, kl_train_dataset, base_dict, stats, gpu_list[i % len(gpu_list)], index_queue, result_queue))
               for i in range(args.workers)]

    del net

    for w in workers:
        w.daemon = True
        w.start()

    max_num = args.max_num if args.max_num > 0 else len(dataset)

    for i in range(max_num):
        index_queue.put(i)

    proc_start_time = time.time()
    if args.save_base_out is not None: base_out_f = open(args.save_base_out, 'wb')
    if args.save_raw_scores is not None: raw_score_f = open(args.save_raw_scores, 'wb')
    out_dict = {}

    for i in range(max_num):
        vid_id, rst, output, base_out = result_queue.get()
        vid_id = vid_id.split('/')[-1]
        if args.save_base_out is not None: pickle.dump((vid_id, base_out), base_out_f, pickle.HIGHEST_PROTOCOL)
        if args.save_raw_scores is not None: pickle.dump((vid_id, output), raw_score_f, pickle.HIGHEST_PROTOCOL)
        out_dict[vid_id] = rst

        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {:.04f} sec/video'.format(i, i + 1,
                                                                        max_num,
                                                                        float(cnt_time) / (i + 1)))
    if args.save_base_out is not None: base_out_f.close()
    if args.save_raw_scores is not None: raw_score_f.close()

    if args.save_scores is not None:
        save_dict = {k: v for k,v in out_dict.items()}
        pickle.dump(save_dict, open(args.save_scores, 'wb'), pickle.HIGHEST_PROTOCOL)
