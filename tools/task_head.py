"""
Script to train task head separately
"""

import pickle
import sys
import argparse
import numpy as np
import datetime
import json
import os
from scipy.special import softmax

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
sys.path.append(os.getcwd())
from ssn_models import GLCU, TC

parser = argparse.ArgumentParser(description="PyTorch code to train task head (GLCU)")
parser.add_argument('train_pkl', type=str, help='Path to the training pickle file')
parser.add_argument('val_pkl', type=str, help='Path to the val pickle file')
parser.add_argument('train_tag', type=str, help='Path to the training TAG file')
parser.add_argument('val_tag', type=str, help='Path to the val TAG file')
parser.add_argument('W_pth', type=str, help='Path of the W matrix')
parser.add_argument('--num_steps', type=int, default=779, help='Number of Steps')
parser.add_argument('--num_tasks', type=int, default=180, help='Number of Tasks')
parser.add_argument('--model_dir', '--md', type=str, help='Path where to save trained models', default=None)
parser.add_argument('--test', default=False, action='store_true', help='Test mode')
parser.add_argument('--model_checkpoint', type=str, help='Path of model checkpoint (required for test mode)')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', '--e', type=int, default=100)
parser.add_argument('--eval-freq', '--ef', type=int, default=10)
parser.add_argument('--batch-size', '-b', type=int, default=16)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--print-freq', '-p', default=1, type=int)
args = parser.parse_args()


class TaskDataSet(Dataset):
    def __init__(self, pkl_file, tag_file):
        super(Dataset, self).__init__()

        self.tag_file = tag_file
        self.pkl_file = pkl_file
        
        # Read dataset
        gt = {vid_id:task_id for vid_id, task_id in self._read_block()}

        self.data = [
            (data, [gt[vid_id]]) for vid_id, data in self._read_pkl().items()
        ]

    def _read_block (self):
        # Read TAG blocks
        f = open(self.tag_file, 'r')

        while (len(f.readline()) > 0):
            # Keep reading the block
            vid_id = f.readline().strip().split('/')[-1] # Vid ID
            f.readline() # Num Frames
            task_id = int(f.readline().strip()) # Task ID
            f.readline() # Useless

            corrects = int(f.readline().strip()) # Num correct
            for _c in range(corrects):
                f.readline()

            preds = int(f.readline().strip())
            for _p in range(preds):
                f.readline()

            yield vid_id, task_id

    def _read_pkl(self):
        # Read the input data
        pkl_data_raw = pickle.load(open(self.pkl_file, 'rb'))
        pkl_data = dict()
        for vid_id, data in pkl_data_raw.items():
            vid_id = vid_id.split('/')[-1]
            act_scores = data[1]
            comp_scores = data[2]
            combined_scores = softmax(act_scores[:, 1:], axis=1) * np.exp(comp_scores)
            pkl_data[vid_id] = combined_scores.mean(axis=0)
        
        return pkl_data
    
    def __getitem__(self, index):
        """
        Get the item at the 'index' position
        """
        input, label = self.data[index]
        return torch.FloatTensor(input), torch.LongTensor(label)

    def __len__(self):
        return len(self.data)


class Trainer:
    """
    Trainer for the class head
    """
    def __init__(self):
        self.net = GLCU(args.num_steps, args.num_tasks, half_unit=True).cuda()
        W = np.load(args.W_pth)
        assert W.shape == (args.num_steps, args.num_tasks)
        self.tc_net = TC(W)

    def train(self):
        """
        Train the net
        """

        train_dataset = TaskDataSet(args.train_pkl, args.train_tag)
        val_dataset = TaskDataSet(args.val_pkl, args.val_tag)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)

        optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum)
        criterion = torch.nn.CrossEntropyLoss().cuda()

        print ('Starting Traning...')
        for epoch in range(1, args.epochs + 1):
            # switch to train mode
            self.net.train()
            tot_loss = 0.0
            for inputs, target in train_loader:
                inputs = torch.autograd.Variable(inputs.cuda())
                target = torch.autograd.Variable(target.squeeze().cuda())
                preds = self.net(inputs)

                loss = criterion(preds, target)
                tot_loss += float(loss.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss /= len(train_dataset)
            torch.cuda.empty_cache()

            timestamp = str(datetime.datetime.now()).split('.')[0]
            if epoch % args.print_freq == 0:
                # print log
                log = json.dumps({
                    'timestamp': timestamp,
                    'epoch': epoch,
                    'train_loss': float('%.5f' % tot_loss),
                    'lr': float('%.6f' % args.lr)
                })
                print (log)
                model_path = os.path.join(args.model_dir, 'epoch_%d.pth'%(epoch))
                self.save_model(model_path)

            # Running on val set
            if epoch % args.eval_freq == 0:
                self.net.eval()
                val_loss = 0.0
                acc_meter = AverageMeter()
                for inputs, target in val_loader:
                    inputs = torch.autograd.Variable(inputs.cuda(), volatile=True)
                    target = torch.autograd.Variable(target.squeeze().cuda(), volatile=True)
                    preds = self.net(inputs)

                    loss = criterion(preds, target)
                    val_loss += float(loss.data)
                    acc = self.compute_acc(preds.data.cpu().numpy(), target.data.cpu().numpy())
                    acc_meter.update(acc, target.size(0))

                val_loss /= len(val_dataset)

                timestamp = str(datetime.datetime.now()).split('.')[0]
                log = json.dumps({
                    'timestamp': timestamp,
                    'epoch': epoch,
                    'val_loss': float('%.5f' % val_loss),
                    'val_acc': float('%.4f' % acc_meter.avg)
                })
                print (log)

    def compute_acc(self, preds, target):
        preds = np.argmax(preds, axis=1)
        assert preds.shape == target.shape
        acc = sum(preds == target) / len(target)
        return acc * 100

    def compute_tc_acc(self):
        val_dataset = TaskDataSet(args.val_pkl, args.val_tag)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
        acc_meter = AverageMeter()
        for inputs, target in val_loader:
            inputs = torch.autograd.Variable(inputs.cuda(), volatile=True)
            target = torch.autograd.Variable(target.squeeze().cuda(), volatile=True)
            tc_preds = self.tc_net(inputs)

            acc = self.compute_acc(tc_preds.data.cpu().numpy(), target.data.cpu().numpy())
            acc_meter.update(acc, target.size(0))

        print ('TC Accuracy: %.4f' % acc_meter.avg)

    def inference(self, model_checkpoint):
        self.load_model(model_checkpoint)
        train_dataset = TaskDataSet(args.train_pkl, args.train_tag)
        val_dataset = TaskDataSet(args.val_pkl, args.val_tag)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)

        train_acc = AverageMeter()
        train_acc_tc = AverageMeter()
        for inputs, target in train_loader:
            inputs = torch.autograd.Variable(inputs.cuda(), volatile=True)
            target = torch.autograd.Variable(target.squeeze().cuda(), volatile=True)
            preds = self.net(inputs)
            tc_preds = self.tc_net(inputs)
            acc = self.compute_acc(preds.data.cpu().numpy(), target.data.cpu().numpy())
            acc_tc = self.compute_acc(tc_preds.data.cpu().numpy(), target.data.cpu().numpy())
            train_acc.update(acc, target.size(0))
            train_acc_tc.update(acc_tc, target.size(0))

        val_acc = AverageMeter()
        val_acc_tc = AverageMeter()
        for inputs, target in val_loader:
            inputs = torch.autograd.Variable(inputs.cuda(), volatile=True)
            target = torch.autograd.Variable(target.squeeze().cuda(), volatile=True)
            preds = self.net(inputs)
            tc_preds = self.tc_net(inputs)
            acc = self.compute_acc(preds.data.cpu().numpy(), target.data.cpu().numpy())
            acc_tc = self.compute_acc(tc_preds.data.cpu().numpy(), target.data.cpu().numpy())
            val_acc.update(acc, target.size(0))
            val_acc_tc.update(acc_tc, target.size(0))
        
        print (('Train Acc.: %.4f\n'
                'Train Acc. (TC): %.4f\n'
                'Val Acc.: %.4f\n'
                'Val Acc. (TC): %.4f') % 
                (train_acc.avg, train_acc_tc.avg, val_acc.avg, val_acc_tc.avg))
        

    def save_model(self, checkpoint_path, model=None):
        if model is None: model = self.net
        torch.save(model.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path, model=None):
        print ("Loading model from %s" % checkpoint_path)
        if model is None: model = self.net
        model.load_state_dict(torch.load(checkpoint_path))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    trainer = Trainer()
    if args.test:
        # Run inference
        trainer.inference(args.model_checkpoint)
    else:
        # Train model
        assert args.model_dir is not None
        trainer.compute_tc_acc()
        trainer.train()
    