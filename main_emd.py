from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
torch.backends.cudnn.bencmark = True
from data_loader_emd import *
import os, sys, random, datetime, time
from os.path import isdir, isfile, isdir, join, dirname, abspath
import argparse
import numpy as np
from scipy import stats
from PIL import Image
from scipy.io import savemat
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vltools import Logger

parser = argparse.ArgumentParser(description='PyTorch Implementation of HED.')
parser.add_argument('--bs', type=int, help='batch size', default=300)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=1e-2)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=8)
parser.add_argument('--gamma', type=float, help='gamma', default=0.1)
parser.add_argument('--wd', type=float, help='weight decay', default=5e-4)
parser.add_argument('--reg', type=float, help='coefficient of regularization', default=10)
parser.add_argument('--maxepoch', type=int, help='max epoch', default=1000)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
parser.add_argument('--save_freq', type=int, help='save frequency', default=50)
parser.add_argument('--emd', type=int, help='emd or cross entropy', default=1)
parser.add_argument('--tree', type=int, help='use decision tree or not', default=1)
parser.add_argument('--cuda', type=str, help='cuda', default='0')
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--checkpoint', type=str, help='checkpoint prefix', default=None)
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# datasets
parser.add_argument('--tmp', type=str, default='tmp', help='root of saving images')
parser.add_argument('--casia', type=str, help='root folder of CASIA-WebFace dataset', 
                    default="/home/guest/xg/database/face-112x112/")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

TMP_DIR = "/media/data1/age_data/"+args.tmp
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
log = Logger(TMP_DIR+'/log.txt')

if args.tree:
    if args.emd:
        args.save_folder = '/media/data1/age_data/parameters_tree_emd/bifurcated_face_recognition_network'
    else:
        args.save_folder = '/media/data1/age_data/parameters_tree_ce/bifurcated_face_recognition_network'
else:
    if args.emd:
        args.save_folder = '/media/data1/age_data/parameters_linear_emd/bifurcated_face_recognition_network'
    else:
        args.save_folder = '/media/data1/age_data/parameters_linear_ce/bifurcated_face_recognition_network'

training_dataset = UnionGenerator(txt_path='sub_tasks/union-classification-resnet/name_age_train.txt', 
                         img_dir='data/CACD2000-aligned/')
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=args.bs, 
                                          shuffle=True, num_workers=0, pin_memory=False)
testing_dataset = UnionGenerator(txt_path='sub_tasks/union-classification-resnet/name_age_test.txt', 
                         img_dir='data/CACD2000-aligned/')
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=args.bs, 
                                          shuffle=False, num_workers=0, pin_memory=False)

from model_emd import Baseline, VGG16_Baseline, VGG16_dt_Baseline

if args.tree:    
    model = nn.DataParallel(VGG16_dt_Baseline(num_age=55)).cuda()
else:
    model = nn.DataParallel(VGG16_Baseline(num_age=55)).cuda()
    
if args.checkpoint:
    model.load_state_dict(torch.load(args.save_folder+'_parameters_'+args.checkpoint+'_0004033.pth'))

weights = []
pi = []
for name, p in model.named_parameters():
    if 'pi' in name:
        pi.append(p)
    else:
        weights.append(p)

optimizer = torch.optim.SGD([{"params": pi, "weight_decay": 0},
                             {"params": weights}], lr=args.lr, weight_decay=args.wd, momentum=args.momentum) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

train_record = {'loss': [], 'mae': [], 'acc': []}
test_record = {'loss': [], 'mae': [], 'acc': []}

def train_epoch(train_loader, model, optimizer, epoch):
    losses = AverageMeter()
    MAE = AverageMeter()
    ACC = AverageMeter()
    batch_time = AverageMeter()
    os.makedirs("/media/data1/age_data/"+args.tmp+"/epoch%d_training"%epoch, exist_ok=True)
    
    model.train()
    if args.emd:
        wloss = WassersteinLoss.apply  
        
        idx = torch.arange(55).repeat(55, 1).cuda()
        idx2 = torch.arange(55).unsqueeze(1).cuda()
        gm = torch.abs(idx - idx2).float()
    else:
        criterion = nn.NLLLoss() 
  
    for batch_idx, (data, age) in enumerate(train_loader):
        start_time = time.time()
        age = torch.squeeze(age)
        prob = model(data)
        
        if args.emd:        
            loss = wloss(prob, age, gm, args.reg)
        else:
            log_prob = torch.log(prob)
            loss = criterion(log_prob, age-11)
    
        _, idx = prob.topk(1, 1, True, True)
        idx = idx.reshape(-1) + 11
        err = torch.abs(idx - age)
        mae = torch.sum(err)
        acc = torch.sum(err <= 5)
        
        losses.update(loss.item(), data.size(0))
        MAE.update(mae.item() / data.size(0), data.size(0))
        ACC.update(acc.item() / data.size(0) * 100, data.size(0))
     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        batch_time.update(time.time() - start_time)
    
        if batch_idx % args.print_freq == 0:
            log.info("Tr|Ep %03d Bt %03d/%03d, (sec/bt: %.2fsec): loss=%.3f (avg=%.3f), mae=%.3f (avg=%.3f), acc=%.3f (avg=%.3f)" \
            % (epoch, batch_idx, len(train_loader), batch_time.val, losses.val, losses.avg, MAE.val, MAE.avg, ACC.val, ACC.avg))
            
            train_record['loss'].append(losses.avg)
            train_record['mae'].append(MAE.avg)
            train_record['acc'].append(ACC.avg)
        
        if batch_idx % args.save_freq == 0:
            idx = np.random.choice(args.bs, 5, replace=False)
            for i in idx:
                pred = prob[i, :].cpu().detach().numpy()
                y_int = int(age[i].item())
                plt.figure()
                plt.bar(range(11, 66), pred, color='yellow')
                plt.bar([y_int], [pred[y_int - 11]], color='blue')
                plt.title('gt: %d'%(y_int))
                plt.savefig("/media/data1/age_data/"+args.tmp+"/epoch%d_training/iter%d_sample%d"%(epoch, batch_idx, i))
                plt.close()
        
        if batch_idx == len(train_loader) - 1: 
            torch.save(model.state_dict(), '{}_parameters_{:03d}_{:07d}.pth'\
                 .format(args.save_folder, epoch, batch_idx+1))
            log.info('checkpoint has been created!')

def test_epoch(test_loader, model, epoch):
    losses = AverageMeter()
    MAE = AverageMeter()
    ACC = AverageMeter()
    batch_time = AverageMeter()
    os.makedirs("/media/data1/age_data/"+args.tmp+"/epoch%d_testing"%epoch, exist_ok=True)
    
    wloss = WassersteinLoss.apply        
    model.eval()    
              
    idx = torch.arange(55).repeat(55, 1).cuda()
    idx2 = torch.arange(55).unsqueeze(1).cuda()
    gm = torch.abs(idx - idx2).float()
    
    if args.emd:
        wloss = WassersteinLoss.apply  
        
        idx = torch.arange(55).repeat(55, 1).cuda()
        idx2 = torch.arange(55).unsqueeze(1).cuda()
        gm = torch.abs(idx - idx2).float()
    else:
        criterion = nn.NLLLoss() 
    
    with torch.no_grad():
        for batch_idx, (data, age) in enumerate(test_loader):
            start_time = time.time()
            age = torch.squeeze(age)
            prob = model(data)

            if args.emd:        
                loss = wloss(prob, age, gm, args.reg)
            else:
                log_prob = torch.log(prob)
                loss = criterion(log_prob, age-11)
        
            _, idx = prob.topk(1, 1, True, True)
            idx = idx.reshape(-1) + 11
            err = torch.abs(idx - age)
            mae = torch.sum(err)
            acc = torch.sum(err <= 5)
            
            losses.update(loss.item(), data.size(0))
            MAE.update(mae.item() / data.size(0), data.size(0))
            ACC.update(acc.item() / data.size(0) * 100, data.size(0))
            batch_time.update(time.time() - start_time)
        
            if batch_idx % args.print_freq == 0:
                log.info("Te|Ep %03d Bt %03d/%03d, (sec/bt: %.2fsec): loss=%.3f (avg=%.3f), mae=%.3f (avg=%.3f), acc=%.3f (avg=%.3f)" \
                % (epoch, batch_idx, len(test_loader), batch_time.val, losses.val, losses.avg, MAE.val, MAE.avg, ACC.val, ACC.avg))
            
            if batch_idx % args.save_freq == 0:
                rn = np.random.choice(data.size(0), 5, replace=False)
                for i in rn:
                    pred = prob[i, :].cpu().detach().numpy()
                    y_int = int(age[i].item())
                    plt.figure()
                    plt.bar(range(11, 66), pred, color='yellow')
                    plt.bar([y_int], [pred[y_int - 11]], color='blue')
                    plt.title('gt: %d'%(y_int))
                    plt.savefig("/media/data1/age_data/"+args.tmp+"/epoch%d_testing/iter%d_sample%d"%(epoch, batch_idx, i))
                    plt.close()
        
    test_record['loss'].append(losses.avg)
    test_record['mae'].append(MAE.avg)
    test_record['acc'].append(ACC.avg)
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, maxnum=16000):
        self.reset()
        self.maxnum = maxnum

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count >= self.maxnum:
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
'''        
class WassersteinLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, label, M, reg, numItermax=100, eps=1e-6):
        # Generate target matrix
        bs = prediction.size(0)
        dim = prediction.size(1)
        
        target = torch.zeros(bs, dim).cuda()
        idx = torch.arange(bs).cuda()
        target[idx, label - 11] = 1                
        
        # Compute Wasserstein Distance
        u = torch.ones(bs, dim, dtype=M.dtype).cuda() / dim
        v = torch.ones(bs, dim, dtype=M.dtype).cuda() / dim
        
        #K= torch.exp((-M/reg)-1)
        K = torch.empty(M.shape, dtype=M.dtype).cuda()
        torch.div(M, -reg, out=K)
        K = K - 1
        torch.exp(K, out=K)
        
        #KM= K * M
        KM = torch.mul(K, M)
        
        #KlogK = K * logK
        KlogK = torch.mul(K, torch.log(K))    

        for i in range(numItermax):
            v = torch.div(target, torch.mm(u, K))
            u = torch.div(prediction, torch.mm(v, K.transpose(0, 1)))
            
        u[torch.abs(u) < eps] = eps
        v[torch.abs(v) < eps] = eps
            
        tmp1 = torch.mm(u, KM)
        loss = torch.mul(v, tmp1).sum()
        
        ulogu = torch.mul(u, torch.log(u))
        tmp2 = torch.mm(ulogu, K)
        entropy1 = torch.mul(tmp2, v).sum()

        vlogv = torch.mul(v, torch.log(v))
        tmp3 = torch.mm(vlogv, K.transpose(0, 1))
        entropy2 = torch.mul(tmp3, u).sum()

        tmp4 = torch.mm(u, KlogK)
        entropy3 = torch.mul(tmp4, v).sum()

        entropy = (entropy1 + entropy2 + entropy3) * reg
        loss_total = (loss + entropy)
            
        # Save intermediate variables
        ctx.save_for_backward(u, torch.tensor([reg], dtype=M.dtype).cuda())
        return loss_total.clone() / args.bs
    
    @staticmethod    
    def backward(ctx, grad_output):
        u, reg = ctx.saved_tensors
        dim = u.size(1)
        grad_input = grad_output.clone()
        
        grad = torch.log(u) 
        shifting = torch.sum(grad, dim=1, keepdim=True) / dim

        return grad_input * (grad - shifting) * reg, None, None, None, None, None

def main():
    test_epoch(testing_dataloader, model, 0)
    for epoch in range(args.maxepoch):
        scheduler.step() # will adjust learning rate
        train_epoch(training_dataloader, model, optimizer, epoch+1)
        test_epoch(testing_dataloader, model, epoch+1)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].plot(train_record['loss'])
        axes[0].legend(['Loss'], loc="upper right")
        axes[0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[0].set_xlabel("Iter")
        axes[0].set_ylabel("Loss")

        axes[1].plot(test_record['mae'])
        axes[1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[1].legend(["MAE"], loc="upper right")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")

        axes[2].plot(test_record['acc'])
        axes[2].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[2].legend(["Top5 acc"], loc="lower right")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Top5 Accuracy")

        plt.tight_layout()
        plt.savefig(TMP_DIR+'/record.pdf')
        plt.close(fig)
        
        
        
if __name__ == '__main__':
    main()
