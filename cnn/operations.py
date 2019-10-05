import torch
import torch.nn as nn

class avg_pool(nn.Module):
  def __init__(self, C, stride, affine):
    self.op = nn.AvgPool2d((1, kernel_size), stride = 1, padding=(0, 1), count_include_pad=False)
  def forward(self, x):
    out = self.op(x)
    return self.op(x)

class max_pool(nn.Module):
  def __init__(self, C, stride, affine):
    self.op = nn.MaxPool2d((1, kernel_size), stride = 1, padding=(0, 1), count_include_pad=False)
  def forward(self, x):
    out = self.op(x)
    return self.op(x)

def avg_pool_wrap(kernel_size):
    #print('AvgPool kernel size: {}'.format(kernel_size))
    return lambda C, stride, affine: nn.AvgPool2d((1, kernel_size), stride = stride, padding=(0, 1))

def max_pool_wrap(kernel_size):
    #print('MaxPool kernel size: {}'.format(kernel_size))
    return lambda C, stride, affine: nn.MaxPool2d((1, kernel_size), stride = stride, padding=(0, 1))

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_1x2' : avg_pool_wrap(3),
  'max_pool_1x2' : max_pool_wrap(3),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_1x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, (0, 1), affine=affine),
  'sep_conv_1x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, (0, 2), affine=affine),
  'sep_conv_1x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, (0, 3), affine=affine),
  'dil_conv_1x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, (0, 2), 2, affine=affine),
  'dil_conv_1x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, (0, 4), 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

def ops_lookup(x):
  c = type(x).__name__
  if c == 'AvgPool2d':
    if c.kernel_size == 3:
      return 'avg_pool_3x3'
    else:
      return 'avg_pool_2x2'
  elif c == 'MaxPool2d':
    if c.kernel_size == 3:
      return 'max_pool_3x3'
    else:
      return 'max_pool_2x2'
  elif c == 'Identity' or c == 'FactorizedReduce':
    return 'skip_connect'
  elif c == 'SepConv':
    if c.kernel_size == 3:
      return 'sep_conv_3x3'
    elif c.kernel_size == 5:
      return 'sep_conv_5x5'
    else:
      return 'sep_conv_7x7'
  elif c == 'DilConv':
    if c.kernel_size == 3:
      return 'dil_conv_3x3'
    elif c.kernel_size == 5:
      return 'dil_conv_5x5'
    else:
      return 'dil_conv_7x7'
  elif c == 'Sequential':
    return 'conv_7x1_1x7'
  elif c == 'Zero':
    return 'none'

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, (1, kernel_size), stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
    #print("ReLUConvBN.init {}".format(kernel_size)) #delete print

  def forward(self, x):
    #print("ReLUConvBN.forward {}".format(x.shape)) #print
    out = self.op(x)
    #print("ReLUConvBN.result {}".format(out.shape))
    return out


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.kernel_size = kernel_size
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    #print("DilConv kernel size: {}".format(kernel_size)) #print

  def forward(self, x):
    #print("DilConv.forward {}".format(x.shape)) #print
    out = self.op(x)
    #print("DilConv.result {}".format(out.shape))
    return out


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.kernel_size = kernel_size
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    #print("SepConv kernel size: {}".format(kernel_size)) #print

  def forward(self, x):
    #print("SepConv.forward {}".format(x.shape)) #print
    out = self.op(x)
    #print("SepConv.result {}".format(out.shape))
    return out


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    #print("Identity.forward {}".format(x.shape))
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
    #print('Zero')

  def forward(self, x):
    #print("Zero data dimension {}".format(x.shape)) #print
    if self.stride == 1:
      out = x.mul(0.)
      return out
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=(1, 2), padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=(1, 2), padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    #print('FactorizedReduce')

  def forward(self, x):
    #print("FactorizedReduce.forward {}".format(x.shape))
    x = self.relu(x)
    # correction of dimension mismatch error
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,:,:])], dim=1)
    out = self.bn(out)
    #print("FactorizedReduce.result {}".format(out.shape))
    #print("FactorizedReduce data dimension {}".format(x.shape)) #print
    return out

class PreprocessReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(PreprocessReduce, self).__init__()
    assert C_out % 2 == 0
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, (1, 1), stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, (1, 1), stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    #print('FactorizedReduce')

  def forward(self, x):
    # correction of dimension mismatch error
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,:,:])], dim=1)
    out = self.bn(out)
    #print('PreprocessReduce: {}'.format(out.shape))
    #print("FactorizedReduce data dimension {}".format(x.shape)) #print
    return out

class PreReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, affine = True):
    super(PreReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, (1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
    #print("ReLUConvBN.init {}".format(kernel_size)) #delete print

  def forward(self, x):
    #print("ReLUConvBN.forward {}".format(x.shape)) #print
    #print(x.shape)
    #print(self.op[1].padding)
    res = self.op(x)
    
    #print('PrePreLUCOnvBN: {}'.format(res.shape))
    return res
