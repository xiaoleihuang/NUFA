"""
    This script is to convert all words into indices
"""
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def write2f(datap, tkn, suffix):
    with open('../data_indices/' + datap + '/' + datap + '.' + suffix, 'w') as wfile:
        with open('../data/' + datap + '/' + datap + '.' + suffix) as dfile:
            wfile.write(dfile.readline()) # column names
            
            tmp_docs = []
            count = 0
            for line in dfile:
                tmp_docs.append(line.strip().split('\t'))
                count += 1
                
                if count % 1000 == 0:
                    texts = [item[1] for item in tmp_docs]
                    indices = pad_sequences(tkn.texts_to_sequences(texts), maxlen=50)

                    for idx in range(len(tmp_docs)):
                        tmp_docs[idx][1] = ' '.join(map(str, indices[idx]))
                        wfile.write('\t'.join(tmp_docs[idx]) + '\n')
                    count = 0
                    tmp_docs = []

            if len(tmp_docs) > 0:
                texts = [item[1] for item in tmp_docs]
                indices = pad_sequences(tkn.texts_to_sequences(texts), maxlen=50)

                for idx in range(len(tmp_docs)):
                    tmp_docs[idx][1] = ' '.join(map(str, indices[idx]))
                    wfile.write('\t'.join(tmp_docs[idx]) + '\n')
                tmp_docs = []    


def data2indices(datap):
    print('Working on: ', datap)

    # load tokenizer 
    tkn_path = './tokenizer/' + datap + '.tkn'
    with open(tkn_path, 'rb') as tkn_file:
        tkn = pickle.load(tkn_file)

    # load the training data
    write2f(datap, tkn, 'train')
    # load the valid data
    write2f(datap, tkn, 'dev')
    # load the testing data
    write2f(datap, tkn, 'test')

data_list = [
    'twitter',
    'amazon',
    'yelp_hotel',
    'yelp_rest',
]

for datap in data_list:
    data2indices(datap)
