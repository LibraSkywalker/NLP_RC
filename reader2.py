import os
import sys
import pickle
import re
import linecache
from collections import Counter
import tensorflow as tf

def counts():
	cache = 'counter.pickle'
	if os.path.exists(cache):
		with open(cache, 'r') as f:
			return pickle.load(f)

	files = ['data/train.txt', 'data/validation.txt', 'data/test.txt']
	counter = Counter()
	for file_name in files:
		document = []
		for line in linecache.getlines(file_name):
			words = re.split(" |\t",line)
			if words[0] == "21":
				if "test" in file_name :
					answer = ["null"]
					query = words[1:-2]
				else:
					answer = [words[-3]]
		 			query = words[1:-3]
				for token in document + query + answer:
					counter[token.lower()] += 1
				document = []
			else:
				document += words[1:-1]
				
	with open(cache, 'w') as f:
		pickle.dump(counter, f)

	return counter

def tokenize(index, word):

	files = ['data/train.txt', 'data/validation.txt', 'data/test.txt']
	counter = Counter()
	for file_name in files:
		out_name = re.split("/|\.",file_name)[1] + ".tfrecords"
		print(out_name)
		writer = tf.python_io.TFRecordWriter(out_name)
		document = []
		for line in linecache.getlines(file_name):
			words = re.split(" |\t",line)
			if words[0] == "21":
				if "test" in file_name :
					answer = ["null"]
					query = words[1:-2]
				else:
					answer = [words[-3]]
		 			query = words[1:-3]
		 		document = map(lambda x : index[x.lower()],document)
		 		query = map(lambda x : index[x.lower()],query)
		 		answer = map(lambda x : index[x.lower()],answer)
		 		example = tf.train.Example(
					features = tf.train.Features(
			 			feature = {
			   				'document': tf.train.Feature(
							int64_list=tf.train.Int64List(value=document)),
			   				'query': tf.train.Feature(
				 			int64_list=tf.train.Int64List(value=query)),
			   				'answer': tf.train.Feature(
				 			int64_list=tf.train.Int64List(value=answer))
			   			}))
			   	serialized = example.SerializeToString()
			   	writer.write(serialized)
			   	document = []
			else:
				document += words[1:-1]
				
def main():
  counter = counts()
  print('num words',len(counter))
  word = sorted(counts, key=counts.get, reverse=True)
  index = {token: i for i, token in enumerate(word)}
  tokenize(index, word)
  print('DONE')

if __name__ == "__main__":
  main()
