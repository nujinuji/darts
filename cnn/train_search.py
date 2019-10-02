import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pandas as pd 
import matplotlib.pyplot as plt

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from visualize import *

from scipy.stats import pearsonr


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# binary classification of binding or not binding
CIFAR_CLASSES = 2


class BindingDataset(torch.utils.data.Dataset):

  def loader(self, annofile, seqfile):
    """Load data from given path

    Parameters
    ----------
    annofile : str
      file name of annotation file. format: <seq> <struct matrix>

    seqfile : str
      file name of sequence file. format: <bind score> <seq>

    Returns
    -------
    tuple of numpy array and str
      2d data matrix and its label
    """
    res = []
    affinity = []
    with open(annofile) as anno_file:
      with open(seqfile) as seq_file:
        for seq_line in seq_file:
          
          # Get affinity and sequence; then transform sequence to numpy array of characters  
          affinity.append(float(seq_line.split(' ')[0]))
          seq = anno_file.readline()[1:].strip()
          seq_numpy = np.array(list(seq))

          # Create empty 2D matrix and fill with one-hot representations
          mat = np.zeros((9, MAX_LEN))
          for i, c in enumerate('ATCG'):
            mat[i, :len(seq)] = (seq_numpy == c).astype(float)

          # Append profile
          for i in range(4, 9):
            profile = anno_file.readline().lstrip().strip().split('\t')
            mat[i, :len(seq)] = [float(x) for x in profile]

          res.append([mat])
    return torch.tensor(res, dtype=torch.float), torch.tensor(affinity, dtype=torch.float)

  def __init__(self, annofile, seqfile, transform = None):
    self.dataset, self.labels = self.loader(annofile, seqfile)

  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    return self.dataset[idx, :, :, :], self.labels[idx]

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  
  train_data = BindingDataset(args.train_annofile, args.train_seqfile)
  valid_data = BindingDataset(args.valid_annofile, args.valid_seqfile)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(0.7 * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=2)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    print('-------------------\n|Starting arch %03d|\n-------------------' % epoch)
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  model.train()

  total_logits = []
  total_target = []

  for step, (input, target) in enumerate(train_queue):

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target.reshape((len(target), 1))).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))

    # input_search dimension: 1 * n * 9 * 41
    # retrieve input_search data then transform to n * 1 * 9 * 41
    input_search = input_search[0]
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    logits_list = [float(x) for x in logits.data.cpu().flatten()]
    target_list = [float(x) for x in target.data.cpu().flatten()]
    pearson, _ = pearsonr(logits_list, target_list)
    total_logits.extend(logits_list)
    total_target.extend(target_list)

    if step % args.report_freq == 0:
      logging.info('batch %03d - training loss: %f, acc: %f', step, loss.data.item(), pearson)

  total_pearson, _ = pearsonr(total_logits, total_target)
  print('----------\nend of epoch - training accuracy: %f\n----------' % total_pearson)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  total_logits = []
  total_target = []

  for step, (input, target) in enumerate(valid_queue):

    input = Variable(input, volatile=True).cuda()
    target = Variable(target.reshape((len(target), 1)), volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    logits_list = [float(x) for x in logits.data.cpu().flatten()]
    target_list = [float(x) for x in target.data.cpu().flatten()]
    pearson, _ = pearsonr(logits_list, target_list)

    total_logits.extend(logits_list)
    total_target.extend(target_list)

    if step % args.report_freq == 0:
      logging.info('epoch %03d - validation loss: %f, acc: %f', step, loss.data.item(), pearson)

  total_pearson, _ = pearsonr(total_logits, total_target)
  print('----------\nend of epoch - valid accuracy: %f\n----------' % total_pearson)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

