# ----------------------------------------------------------------------------
# Preprocessing of clamped RNA sequence data
#
# Specific formatting of input for DARTS
#
# https://github.com/nujinuji/darts
# ----------------------------------------------------------------------------

# python preprocess.py <annotation file> <sequence file> <dest folder>

import os
import sys
import numpy as np


# define binding threshould and maximum sequence length
CUTOFF = 0
MAX_LEN = 41


def write_file(profile, fname):
  with open(fname, 'w') as f:
    f.write(profile)


counter = 0


def zero_padding(holder, length, separator):
  """Make shorter sequences zero-padded

  Parameters
  ----------
  holder : str
    zeros for matrix padding
  length : int
    length that need to pad at the end of shorter sequences
  separator : str
    tab separated

  Returns
  -------
  str
    sequence with zero padding
  """
  return separator.join([holder] * int(length))


# reading clamped annotation files and sequence files from DLPRB
with open(sys.argv[1]) as anno_file:
  with open(sys.argv[2]) as seq_file:
    for seq_line in seq_file:
      affinity = float(seq_line.split(' ')[0])
      
      # label files according to cutoff
      if affinity >= CUTOFF:
        lbl = 'bind'
      else:
        lbl = 'not_bind'

      # generate one hot encoding matrix seperated by tab
      seq = anno_file.readline()[1:].strip()
      A = np.array(list(seq))
      A[A != 'A'] = '0'
      A[A == 'A'] = '1'
      A = A.tolist()
      T = np.array(list(seq))
      T[T != 'T'] = '0'
      T[T == 'T'] = '1'
      T = T.tolist()
      C = np.array(list(seq))
      C[C != 'C'] = '0'
      C[C == 'C'] = '1'
      C = C.tolist()
      G = np.array(list(seq))
      G[G != 'G'] = '0'
      G[G == 'G'] = '1'
      G = G.tolist()
      res = []
      res.append(('%s\t%s' % ('\t'.join(A), zero_padding('0', MAX_LEN - int(len(A)), '\t'))).strip().rstrip())
      res.append(('%s\t%s' % ('\t'.join(T), zero_padding('0', MAX_LEN - int(len(T)), '\t'))).strip().rstrip())
      res.append(('%s\t%s' % ('\t'.join(C), zero_padding('0', MAX_LEN - int(len(C)), '\t'))).strip().rstrip())
      res.append(('%s\t%s' % ('\t'.join(G), zero_padding('0', MAX_LEN - int(len(G)), '\t'))).strip().rstrip())

      # adding structural information
      for i in range(5):
        res.append((anno_file.readline().lstrip().strip() + '\t' + zero_padding('0', MAX_LEN - int(len(A)), '\t')).strip().rstrip())

      # write to file with labels
      if not os.path.exists(os.path.join(sys.argv[3], lbl)):
        os.makedirs(os.path.join(sys.argv[3], lbl))
      write_file('\n'.join(res), os.path.join(sys.argv[3], lbl, str(counter) + '.ext'))
      counter += 1
