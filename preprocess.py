import os
import sys

# python preprocess.py <annotation file> <sequence file> <dest folder>

CUTOFF = 0
MAX_LEN = 41


def write_file(profile, fname):
	with open(fname, 'w') as f:
		f.write(profile)

counter = 0

def zero_padding(holder, length, separator):
	return separator.join([holder] * int(length))

with open(sys.argv[1]) as anno_file:
	with open(sys.argv[2]) as seq_file:
		for seq_line in seq_file:
			affinity = float(seq_line.split(' ')[0])
			
			if affinity >= CUTOFF:
				lbl = 'bind'
			else:
				lbl = 'not_bind'
			
			seq = anno_file.readline()[1:].strip()
			A = np.array(seq)
			A[not A == 'A'] = 0
			A[A == 'A'] = 1
			A = A.tolist()
			T = np.array(seq)
			T[not T == 'T'] = 0
			T[T == 'T'] = 1
			T = T.tolist()
			C = np.array(seq)
			C[not C == 'C'] = 0
			C[C == 'C'] = 1
			C = C.tolist()
			G = np.array(seq)
			G[not G == 'G'] = 0
			G[G == 'G'] = 1
			G = G.tolist()
			res = []
			res.append('%s\t%s' % ('\t'.join(A), zero_padding('0', MAX_LEN - int(len(A)), '\t')))
			res.append('%s\t%s' % ('\t'.join(T), zero_padding('0', MAX_LEN - int(len(T)), '\t')))
			res.append('%s\t%s' % ('\t'.join(C), zero_padding('0', MAX_LEN - int(len(C)), '\t')))
			res.append('%s\t%s' % ('\t'.join(G), zero_padding('0', MAX_LEN - int(len(G)), '\t')))
			for i in range(5):
				res.append(anno_file.readline().lstrip().strip() + '\t' + zero_padding('0', MAX_LEN - int(len(A)), '\t'))
			write_file('\n'.join(res), os.path.join(sys.argv[3], lbl, str(counter) + '.ext'))
			counter += 1
