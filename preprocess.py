#!/usr/bin/env python

import sys
from math import ceil
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

def get(file_name: str, cutoff=None, verbose=1, is_evaluate: bool=False):
	size = 0
	prev = 0

	with open(file_name) as file:
		_, *data = list(file)
		
		for _, line in enumerate(data[:cutoff]):
			size += len(line)
			prev += size

			lst = line.strip().split(',')

			if is_evaluate:
				floats = [float(x) for x in lst[1:-1]]
				categories = [lst[-1]]
				label = None
			else:
				floats = [float(x) for x in lst[1:-2]]
				categories = [lst[-2]]
				label = lst[-1]

			if (prev / 1e6) > 2e6 and verbose:
				print('Dataset read: %.3fMB' % (size / 1e6), end='\r')
				prev = 0

			yield (np.array(floats + categories), label)

	if verbose:
		print('\r', end='')
		print('Dataset read: %.3fMB' % (size / 1e6))


def get_preprocessor(x):
	numeric_features = list(range(len(x[0]) - 1))
	categorical_features = [numeric_features[-1] + 1]

	numeric_transformer = Pipeline(
		steps=[
			('Imputer', SimpleImputer(strategy='median', verbose=1)),
			# ('Scaler', StandardScaler()),
		]
	)

	categorical_transformer = Pipeline(
		steps=[
			('Encoder', OneHotEncoder(handle_unknown='ignore'))
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			('Numerical', numeric_transformer, numeric_features),
			('Categorical', categorical_transformer, categorical_features)
		]
	)

	return preprocessor


def get_dataset(source: str, cutoff=None, is_evaluate: bool = False):
	dataset = list(get(source, cutoff=cutoff, is_evaluate=is_evaluate))

	print('Picking out dataset features ...')
	x = np.array([x for x, _ in dataset])

	if is_evaluate:
		y = None
	else:
		y = np.array([1 if y == 'C' else -1 for _, y in dataset])

	print('Fitting preprocessor transform ...')
	return get_preprocessor(x).fit_transform(x), y


def encode(source: str, target: str, is_evaluate: bool = False):
	print('[Encode] Processing file %s -> %s' % (source, target))

	with open(target, 'w') as file:
		print('Reading dataset ...')
		x, y = get_dataset(source, cutoff=None, is_evaluate=is_evaluate)
		length = len(x)
		logi = ceil(length / 100)
		size = 0

		for i in range(length):
			if is_evaluate:
				addition = ''
			else:
				addition = ',' + str(int(y[i]))

			res = ','.join(map(lambda x: str(x), x[i])) + addition
			size += len(res)

			file.write(res)
			file.write('\n')

			if i % logi == 0 or i == length - 1:
				print('[%.3f%%] %.3fMB' % (i / length * 100, size / 1e6), end='\r')

		print('\r', end='')
		print('Done generating file [%.3fMB]' % (size / 1e6))


def decode(source: str, target: str):
	print('[Decode] Processing file %s -> %s' % (source, target))
	
	with open(target, 'w') as target, open(source) as source:
		print('Converting classification file from {1, -1} to {"C", "L"} ...')

		result = ['C' if float(y) > 0 else 'L' for y in source]
		target.write('\n'.join(result))


def strip_train_class(source: str, target: str):
	with open(source) as file, open(target, 'w') as target_file:
		header, *data = list(file)

		target_file.write(header)
		
		for line in data[:200000]:
			lst = line.strip().split(',')
			rest = [x for x in lst[:-1]]

			target_file.write(','.join(rest))
			target_file.write('\n')
			


if __name__ == '__main__':
	if len(sys.argv) < 4:
		print('Usage: [option] [source] [target]')
		exit(0)

	option = sys.argv[1]
	source = sys.argv[2]
	target = sys.argv[3]

	if option == "encode_evaluation":
		encode(source, target, is_evaluate=True)
	elif option == "encode_train":
		encode(source, target, is_evaluate=False)
	elif option == "decode":
		decode(source, target)
	elif option == "strip_train_class":
		strip_train_class(source, target)