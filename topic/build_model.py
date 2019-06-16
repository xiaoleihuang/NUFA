import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import pickle
from nltk.tokenize import word_tokenize

def run(data_name):
    print('Working on ' + data_name)
    corpus = []

    # preprocess
    with open('../data/' + data_name + '/' + data_name + '.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            corpus.append(line[1].split())

    # build dictionary
    dictionary = Dictionary(corpus)
    dictionary.save(data_name + '.dict')

    # documents to indices
    doc_matrix = [dictionary.doc2bow(doc) for doc in corpus]
    del corpus # release memory
    ldamodel = LdaModel(doc_matrix,
            id2word=dictionary, num_topics=10,
            passes=2, alpha='symmetric', eta=None)
    ldamodel.save(data_name + '.model')


data_list = [
    'twitter',
    'amazon',
    'yelp_hotel',
    'yelp_rest',
]

for data_name in data_list:
    run(data_name)
