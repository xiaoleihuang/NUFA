import pickle
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, accuracy_score
import sklearn


def build_data(data_name, split_prop=0.7):
    """
        split_prop: the proportion of training data
    """
    print('Working on: ' + data_name)
    dkeys = [-5, -4, -3, -2] # -5, -4, -3, -2

    # loop through each domain key
    for dkey in dkeys:
        label_dict = dict()
        doc_dict = dict()

        # load the data
        with open('../data/' + data_name + '/' + data_name + '.tsv') as dfile:
            print('Working on the domain: ' + dfile.readline().strip().split('\t')[dkey])
            for line in dfile:
                line = line.strip().split('\t')
                
                # skip the empty label doc
                if line[dkey] == 'x':
                    continue

                # record the documents and labels
                label = int(line[dkey])
                if label not in label_dict:
                    label_dict[label] = []
                    doc_dict[label] = []

                doc_dict[label].append(line[1])
                label_dict[label].append(label)

        # find the minimum length
        min_len = min([len(label_dict[label]) for label in label_dict])
        # downsample process
        for label in label_dict:
            # skip the label already the minimum number
            if len(label_dict[label]) == min_len:
                continue

            indices = list(range(len(label_dict[label])))
            indices = np.random.choice(indices, size=min_len, replace=False)
            
            label_dict[label] = [label_dict[label][idx] for idx in indices]
            doc_dict[label] = [doc_dict[label][idx] for idx in indices]            

        # merge the docs and labels
        results = dict()
        results['docs'] = []
        results['labels'] = []
        results['vect'] = ''

        for label in label_dict:
            results['labels'].extend(label_dict[label])
            results['docs'].extend(doc_dict[label])

        # release memory
        label_dict = None
        del label_dict
        doc_dict = None
        del doc_dict

        # shuffle the data
        indices = list(range(len(results['labels'])))
        np.random.shuffle(indices)
        results['labels'] = [results['labels'][idx] for idx in indices]
        results['docs'] = [results['docs'][idx] for idx in indices]

        # build the vectorizer
        results['vect'] = TfidfVectorizer(
            ngram_range=(1, 3), 
            min_df=2, max_features=15000
        )
        results['vect'].fit(results['docs'])

        # build train data
        results['train_docs'] = results['docs'][:int(len(results['docs'])*split_prop)]
        results['test_docs'] = results['docs'][int(len(results['docs'])*split_prop):]
        results['train_labels'] = results['labels'][:int(len(results['labels'])*split_prop)]
        results['test_labels'] = results['labels'][int(len(results['labels'])*split_prop):]

        del results['docs']
        del results['labels']

        # save to file
        pickle.dump(results, open('./data/'+data_name+'.'+str(dkey)+'.pkl', 'wb'))


def run_clf(data_name):
    print('Build classifier for: ' + data_name)
    dkeys = [-5, -4, -3, -2]

    # loop through each domain key
    for dkey in dkeys:
        print('Domain key: ', str(dkey))
        # load the data
        results = pickle.load(open('./data/'+data_name+'.'+str(dkey)+'.pkl', 'rb'))
        
        # vectorize train
        results['train_docs'] = results['vect'].transform(results['train_docs'])

        # train classifier
        clf = LogisticRegression(multi_class='ovr')
        clf.fit(results['train_docs'], results['train_labels'])

        # vectorize test
        results['test_docs'] = results['vect'].transform(results['test_docs'])

        # test
        y_preds = clf.predict(results['test_docs'])
        
        # evaluate and record
        with open('results.txt', 'a') as wfile:
            wfile.write(data_name + '\t' + str(dkey) + '_____________\n')
            wfile.write(str(accuracy_score(
                y_pred=y_preds, y_true=results['test_labels']))+'\n'
            )
            wfile.write(str(f1_score(
                y_pred=y_preds, y_true=results['test_labels'],
                average='weighted'))+'\n'
            )
            wfile.write(
                classification_report(y_pred=y_preds, y_true=results['test_labels'], digits=3)+'\n'
            )
            wfile.write('.........................\n')


data_list = [
    'twitter',
    'amazon',
    'yelp_hotel',
    'yelp_rest',
]

for data_name in data_list:
    build_data(data_name, split_prop=0.7)
    run_clf(data_name)
