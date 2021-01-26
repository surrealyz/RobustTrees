import numpy as np
from sklearn import datasets
import pickle
import argparse


parser = argparse.ArgumentParser(description='dump data from libsvm to csv format.')
parser.add_argument('--model_path', type=str, default=None, help='binary xgboost model path.')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--train_method', type=str, default="greedy")
parser.add_argument('--dataset', type=str, default="ijcnn")
parser.add_argument('--num_attacks', type=int, default=5000)
parser.add_argument('--num_classes', type=str, default="2")
parser.add_argument('--attack', action="store_true", default=False)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--train', action="store_true", default=False)
args = parser.parse_args()

dataset = args.dataset
print(args.train)

if args.dataset in ["cod-rna", "higgs"]:
	fstart = 0
else:
	fstart = 1

print("dataset:", dataset)
if dataset == "cod-rna":
	if args.train:
		data_path = "cod-rna_s"
	else:
		data_path = "cod-rna_s.t"
	nfeat = "8"
	args.num_classes = 2
	args.num_attacks = 5000
elif dataset == "binary_mnist":
	if args.train:
		data_path = "binary_mnist0"
	else:
		data_path = "binary_mnist0.t"
	nfeat = "784"
	args.num_classes = 2
	args.num_attacks = 1000
elif dataset == "ijcnn":
	if args.train==True:
		data_path = "ijcnn1s0"
	else:
		data_path = "ijcnn1s0.t"
	nfeat = "22"
	args.num_classes = 2
	args.num_attacks = 5000
elif dataset == "breast_cancer":
	if args.train:
		data_path = "breast_cancer_scale0.train"
	else:
		data_path = "breast_cancer_scale0.test"
	nfeat = "10"
	args.num_classes = 2
	args.num_attacks = 137

elif dataset == "fashion":
	if args.train:
		data_path = "fashion.train0"
	else:
		data_path = "fashion.test0"
	nfeat = "10"
	args.num_classes = 10
	args.num_attacks = 100

else:
	print("no such dataset")
	exit()

print(data_path)
x_test, y_test = datasets.load_svmlight_file(data_path)
x_test = x_test.toarray()
if fstart == 1:
    x_test = np.hstack((np.zeros((x_test.shape[0],fstart)),x_test))
y_test = y_test[:,np.newaxis].astype(int)

if type(x_test) != np.ndarray:
	x_test = x_test.toarray()

if type(y_test) != np.ndarray:
	y_test = y_test.toarray()

print(x_test.shape, y_test.shape)
test_data = np.concatenate([y_test, x_test], axis=1)
print(x_test.shape, y_test.shape, test_data.shape)

if args.train:
	np.savetxt(args.dataset+".train.csv", test_data, fmt='%f', delimiter=',')
else:
	np.savetxt(args.dataset+".test.csv", test_data, fmt='%f', delimiter=',')




