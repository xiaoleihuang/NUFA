"""
    The script is to build features for the baselines
"""
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import sys


# output directory
opt_dir = './vects/'
data_list = [
    'twitter',
    'amazon',
    'yelp_hotel',
    'yelp_rest',
]


def convert(docs, vect_opt):
    # create general vectorizer
    if not os.path.exists(vect_opt):
        vect = TfidfVectorizer(ngram_range=(1,3), max_features=15000, min_df=2)
        vect.fit(docs)
        # save to file
        pickle.dump(vect, open('./vects/' + vect_opt + '.pkl', 'wb'))
    else:
        vect = pickle.load(open(vect_opt, 'rb'))
    return vect


def transform(docs, vect, feas_opt):
    # transform the docs
    feas = vect.transform(docs)
    sparse.save_npz('./feas/' + feas_opt, feas)


def build_vects(data_name):
    print('Working on: ' + data_name)
    # load data
    datap = '../data/' + data_name + '/' + data_name + '.tsv'
    domain_idx = [-5, -4, -3, -2]
    domain_dic = dict()

    with open(datap) as dfile:
        columns = dfile.readline().strip().split('\t')
        docs = []

        for idx, line in enumerate(dfile):
            line = line.strip().split('\t')
            for didx in domain_idx:
                if line[didx] == 'x':
                    continue

                if didx not in domain_dic:
                    domain_dic[didx] = dict()

                if line[didx] not in domain_dic[didx]:
                    domain_dic[didx][line[didx]] = set()
                domain_dic[didx][line[didx]].add(idx)

            docs.append(line[1])
    # general domain
    print('Convert general domain...........')
    vect = convert(docs, '#'.join([data_name, 'general']))
    for suffix in ['.train', '.dev', '.test']:
        tdocs = []
        with open('../data/' + data_name + '/' + data_name + suffix) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')
                tdocs.append(line[1])

        transform(tdocs, vect, '#'.join([data_name, 'general', suffix]))

    print('Convert attributes domain..........')
    # loop through each domain
    for didx in domain_dic:
        for dtype in domain_dic[didx]:
            tmp_docs = [doc for doc_idx, doc in enumerate(docs) if doc_idx in domain_dic[didx][dtype]]
            vect = convert(tmp_docs, '#'.join([data_name, str(didx), str(dtype)]))
            
            for suffix in ['.train', '.dev', '.test']:
                tdocs = []
                with open('../data/' + data_name + '/' + data_name + suffix) as dfile:
                    dfile.readline()
                    for line in dfile:
                        line = line.strip().split('\t')
                        if line[didx] == dtype:
                            tdocs.append(line[1])
                        else:
                            tdocs.append('')
                transform(tdocs, vect, '#'.join([data_name, str(didx), str(dtype), suffix]))

for datap in data_list:
    build_vects(datap)
