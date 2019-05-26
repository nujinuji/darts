import os
import sys

# python preprocess.py <annotation file> <sequence file> <dest folder>

CUTOFF = 5

def write_file(profile, fname):
	with open(fname, 'w') as f:
		f.write(profile)

counter = 0

with open(sys.argv[1]) as anno_file:
	with open(sys.argv[2]) as seq_file:
		for seq_line in seq_file:
			affinity = float(seq_line.split(' ')[0])
			if affinity >= CUTOFF:
				lbl = 'bind'
			else:
				lbl = 'not_bind'
			anno_file.readline()
			res = []
			for i in range(5):
				res.append(anno_file.readline().lstrip().strip())
			write_file('\n'.join(res), os.path.join(sys.argv[3], lbl, str(counter) + '.ext'))
			counter += 1