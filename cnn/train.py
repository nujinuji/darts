import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
from operations import ops_lookup
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from visualize import *

from scipy.stats import pearsonr

if __name__ == '__main__':
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--train_annofile', type=str, default='../dlprb_train/RNCMPT00001.txt.annotations.RNAcontext-sample', help='location of the annotation file for training set')
  parser.add_argument('--train_seqfile', type=str, default='../dlprb_train/RNCMPT00001.txt.sequences.RNAcontext.clamp-sample', help='location of the sequence file for training set')
  parser.add_argument('--valid_annofile', type=str, default='../dlprb_test/RNCMPT00001.txt.annotations.RNAcontext-sample', help='location of the annotation file for validation/testing set')
  parser.add_argument('--valid_seqfile', type=str, default='../dlprb_test/RNCMPT00001.txt.sequences.RNAcontext.clamp-sample', help='location of the sequence file for validation/testing set')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
  parser.add_argument('--layers', type=int, default=2, help='total number of layers')
  parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
  parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
  parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
  args = parser.parse_args()

  args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 2

# define binding threshould and maximum sequence length
CUTOFF = 0
MAX_LEN = 41

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
    '''
    lbl = ['bind', 'not_bind'].index(path.split('/')[-2]) 
    csv = pd.read_csv(path, sep='\t', header = None).values
    
    data = np.zeros((2, 5, 41))
    data[0, :4, :] = csv[:4, :]
    data[1, :, :] = csv[4:, :]
    
    data = csv
    '''
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

def main(args):
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

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.MSELoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_data = BindingDataset(args.train_annofile, args.train_seqfile)
  valid_data = BindingDataset(args.valid_annofile, args.valid_seqfile)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(0.7 * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    print('-------------------\n|Starting arch %03d|\n-------------------' % epoch)
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
    logging.info('train_acc %f', train_acc)

    torch.cuda.empty_cache()

    valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
    logging.info('valid_acc %f', valid_acc)

    plot(list(zip([ops_lookup(x) for x in model.cells[0]._ops], model.cells[0].indices)), "arch_epoch%03d" % epoch)
    utils.save(model, os.path.join(args.save, 'weights.pt'))
  

def pearson_corr(x, y):
  vx = x - torch.mean(x)
  vy = y - torch.mean(y)
  return vx.dot(vy.flatten()) / (torch.norm(vx, 2) * torch.norm(vy, 2))


def train(train_queue, model, criterion, optimizer, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  
  for controller_epoch in range(1):
    total_logits = []
    total_target = []
    for step, (input, target) in enumerate(train_queue):
      input = Variable(input).cuda()
      target = Variable(target.reshape((len(target), 1))).cuda(async=True)

      optimizer.zero_grad()
      logits, logits_aux = model(input)
      loss = criterion(logits, target)
      if args.auxiliary:
        loss_aux = criterion(logits_aux, target)
        loss += args.auxiliary_weight*loss_aux

      loss.backward()
      nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
      optimizer.step()

      logits_list = [float(x) for x in logits.data.cpu().flatten()]
      target_list = [float(x) for x in target.data.cpu().flatten()]
      pearson, _ = pearsonr(logits_list, target_list)
      #pearson_corr(logits.flatten(), target)

      total_logits.extend(logits_list)
      total_target.extend(target_list)
      n = input.size(0)
      if input.size(0) != 48:
        objs.update(loss.data.item(), n)
        top1.update(pearson, n)
      if step % args.report_freq == 0:
        logging.info('batch %03d - training loss: %f, acc: %f', step, loss.data.item(), pearson)
    
      del input
      del target
      torch.cuda.empty_cache()

    total_pearson, _ = pearsonr(total_logits, total_target)
    print('----------\nend of epoch - training accuracy: %f\n----------' % total_pearson)

  plt.clf()
  plt.scatter(total_logits, total_target)
  plt.title('Training relationship - x:logits, y:target')
  plt.savefig(os.path.join(args.save, 'train_%d.png' % epoch), dpi=200)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()

  total_logits = []
  total_target = []

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target.reshape((len(target), 1)), volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)


    logits_list = [float(x) for x in logits.data.cpu().flatten()]
    target_list = [float(x) for x in target.data.cpu().flatten()]
    pearson, _ = pearsonr(logits_list, target_list)
    #pearson_corr(logits.flatten(), target)

    total_logits.extend(logits_list)
    total_target.extend(target_list)
 
    n = input.size(0)
    if input.size(0) != 48:
      objs.update(loss.data.item(), n)
      top1.update(pearson, n)

    if step % args.report_freq == 0:
      logging.info('epoch %03d - validation loss: %f, acc: %f', step, loss.data.item(), pearson)
    
    del input
    del target
    torch.cuda.empty_cache()

  total_pearson, _ = pearsonr(total_logits, total_target)
  print('----------\nend of epoch - valid accuracy: %f\n----------' % total_pearson)

  plt.clf()
  plt.scatter(total_logits, total_target)
  plt.title('Validation relationship - x:logits, y:target')
  plt.savefig(os.path.join(args.save, 'valid_%d.png' % epoch), dpi=200)
  return top1.avg, objs.avg


if __name__ == '__main__':
  main(args) 

