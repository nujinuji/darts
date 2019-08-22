import numpy as np
import pandas as pd 
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
import torch.nn as nn
import sys
import os

t = {'bind': 1, 'not_bind': 0}

class simpleCNN(nn.Module):

    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(6*5*5, 27)
        self.fc2 = nn.Linear(27, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 6*5*5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def loader(path):
  """Load data from given path

  Parameters
  ----------
  path : str
    file directory containing binding information

  Returns
  -------
  tuple of numpy array and str
    2d data matrix and its label
  """
  lbl = ['bind', 'not_bind'].index(path.split('/')[-2]) 
  csv = pd.read_csv(path, sep='\t', header = None).values
  '''
  data = np.zeros((2, 5, 41))
  data[0, :4, :] = csv[:4, :]
  data[1, :, :] = csv[4:, :]
  '''
  data = csv
  return data, lbl


def transform(d):
  """Transform data to tensor from NxHxW to NxCxHxW by adding 1 dimension

  Parameters
  ----------
  d : pytorch.tensor
    3d matrix of input sequence data

  Returns
  -------
  tuple of torch.tensor and str
    4d matrix of float with its label
  """
  data, label = d[0], d[1]
  try:
    return torch.tensor(data, dtype=torch.float32).view(1, 9, 41), label
  except ValueError:
    sys.stderr.write(str(data))
    return 0



def load_data(l):
    res = []
    labels = []
    for i in l:
        label = t[i.split('/')[-2]]
        c = pd.read_csv(i, sep='\t', header = None).values.flatten()
        res.append(c)
        labels.append(label)
    return res, labels

def main(train_path, test_path):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    cudnn.enabled=True

    net = simpleCNN()
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

    print('loading dataset')
    train_data = dset.DatasetFolder(train_path, loader, ['ext'], transform=transform)
    test_data = dset.DatasetFolder(test_path, loader, ['ext'], transform=transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    print('training')
    for epoch in range(100):
        running_loss = 0.0
        train_total, train_correct = 0, 0
        valid_total, valid_correct = 0, 0
        for step, (input, labels) in train_queue:
            input = input[0].cuda()
            optimizer.zero_grad()
            outputs = net(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_total += labels.size[0]
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        for step, (input, labels) in valid_queue:
            input = input[0].cuda()
            outputs = net(input)
            valid_total += labels.size[0]
            _, predicted = torch.max(outputs.data, 1)
            valid_correct += (predicted == labels).sum().item()
        print('at epoch %d: train_acc: %f, test_acc: %f' % (epoch, float(train_correct) / train_total, float(valid_correct) / valid_total))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])