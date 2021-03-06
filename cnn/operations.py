import torch
import torch.nn as nn

def avg_pool_wrap(kernel_size):
    #print('AvgPool kernel size: {}'.format(kernel_size))
    return lambda C, stride, affine: nn.AvgPool2d(kernel_size, stride = stride, padding=1, count_include_pad=False)

def max_pool_wrap(kernel_size):
    #print('MaxPool kernel size: {}'.format(kernel_size))
    return lambda C, stride, affine: nn.MaxPool2d(kernel_size, stride = stride, padding=1)

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : avg_pool_wrap(3),
  'max_pool_3x3' : max_pool_wrap(3),
  'max_pool_2x2' : max_pool_wrap(2),
  'avg_pool_2x2' : avg_pool_wrap(2),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_2x2' : lambda C, stride, affine: SepConv(C, C, 2, stride, 1, affine=affine),
  'dil_conv_2x2' : lambda C, stride, affine: DilConv(C, C, 2, stride, 2, 2, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
    #print("ReLUConvBN.init {}".format(kernel_size)) #delete print

  def forward(self, x):
    #print("ReLUConvBN.forward {}".format(x.shape)) #print
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    #print("DilConv kernel size: {}".format(kernel_size)) #print

  def forward(self, x):
    #print("DilConv data dim: {}".format(x.shape)) #print
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    #print("SepConv kernel size: {}".format(kernel_size)) #print

  def forward(self, x):
    #print("SepConv data dim: {}".format(x.shape)) #print
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
    #print('Zero')

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    print("Zero data dimension {}".format(x.shape)) #print
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    #print('FactorizedReduce')

  def forward(self, x):
    x = self.relu(x)
    # correction of dimension mismatch error
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,:,:])], dim=1)
    out = self.bn(out)
    #print("FactorizedReduce data dimension {}".format(x.shape)) #print
    return out

