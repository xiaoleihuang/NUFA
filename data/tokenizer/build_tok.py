"""
    This function is to create keras tokenizer for each dataset
"""
from keras.preprocessing.text import Tokenizer
import pickle


def build_tok(datap):
    print('Working on: ', datap)
    corpora = list()
    with open('../' + datap+'/'+datap+'.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            corpora.append(line[1])

    print('Fit the corpus')
    tkn = Tokenizer(num_words=15000, split=' ')
    tkn.fit_on_texts(corpora)

    print('Save and finish')
    with open(datap + '.tkn', 'wb') as wfile:
        pickle.dump(tkn, wfile)


data_list = [
    'twitter',
    'amazon',
    'yelp_rest',
    'yelp_hotel',
]

for datap in data_list:
    build_tok(datap)
