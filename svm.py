import numpy as np
import pandas as pd 
from sklearn.svm import SVC
import sys

t = {'bind': 1, 'not_bind': 0}

def load_data(l):
	res = []
	labels = []
	for i in l:
		label = t[i.split('/')[-2]]
		c = pd.read_csv(i, sep='\t', header = None).values.flatten()
		res.append(c)
		labels.append(label)
	return res, label 

def train(x, y):
	svc = SVC()
	svc.fit(x, y)
	return svc 

def test(svc, x, y):
	return svc.score(x, y)

def main(train, test):
	print('loading dataset')
	train_x, train_y = load_data([os.path.join(train, 'bind', x) for x in os.listdir(os.path.join(train, 'bind'))] + [os.path.join(train, 'not_bind', x) for x in os.listdir(os.path.join(train, 'not_bind'))])
	test_x, test_y = load_data([os.path.join(test, 'bind', x) for x in os.listdir(os.path.join(test, 'bind'))] + [os.path.join(test, 'not_bind', x) for x in os.listdir(os.path.join(test, 'not_bind'))])
	print('training')
	c = train(train_x, train_y)
	print('testing')
	train_acc, test_acc = test(c, train_x, train_y), test(c, test_x, test_y)
	print('train_acc: %f, test_acc: %f' % (train_acc, test_acc))

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])