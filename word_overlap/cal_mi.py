import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os


def cal_overlap(data_name, topn = 500):
    print('Working on: ' + data_name)
    fields = [
        ('country', [0, 1], -5), 
        ('region', [0, 1, 2, 3], -4),
        ('gender', [0, 1], -3),
        ('age', [0, 1], -2)]
    
    results = dict()
    
    # loop through each data field
    for field in fields:
        print('Overlap field:', field[0])
        # load the data
        docs = []
        labels = []
        dlabels = []
        with open('../data_hash/' + data_name + '/' + data_name + '.tsv') as dfile:
            cols = dfile.readline().strip().split('\t')
            for line in dfile:
                line = line.strip().split('\t')
                if line[field[-1]] == 'x':
                    continue
        
                docs.append(line[1])
                labels.append(int(line[-1]))
                dlabels.append(int(line[field[-1]]))

        # sample for yelp rest, too large
        if len(docs) > 200000:
            sample_idx = list(range(len(docs)))
            np.random.shuffle(sample_idx)
            sample_idx = sample_idx[:int(0.2*len(sample_idx))]
            docs = [docs[item] for item in sample_idx]
            labels = [labels[item] for item in sample_idx]
            dlabels = [dlabels[item] for item in sample_idx]
        
        print('Build vectorizer....')
        vect = TfidfVectorizer(ngram_range=(1,3), max_features=15000, min_df=2)
        vect.fit(docs)
        docs = np.asarray(docs)
        labels = np.asarray(labels)

        top_dict = dict()
        # loop through the types
        print('Loop through demographic types')
        for dval in field[1]:
            tmp_idxs = [item for item in range(len(dlabels)) if dlabels[item] == dval]
            # select the mi scores for features
            eval_vals = mutual_info_classif(vect.transform(docs[tmp_idxs]), labels[tmp_idxs])
            # use the argmax to find the top feature indices
            top_indices = set(list(np.argsort(eval_vals)[::-1][:topn]))

            # reload the data
            top_dict[dval] = top_indices
    
        # calculate the overlap
        start_key = int(field[1][0])
        overlap = top_dict[start_key]
        for key in field[1][1:]:
            key = int(key)
            overlap = overlap.intersection(top_dict[key])

        results[field[0]] = len(overlap)/topn
        print('Current overlap', field, ':', str(results[field[0]]))

    return results


# visualization
def viz_perform(df, title='default', outpath='./overlap.pdf'):
    """
    Heatmap visualization
    :param df: an instance of pandas DataFrame
    :return:
    """
    a4_dims = (15.7, 12.57)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=1.2)
    viz_plot = sns.heatmap(df, annot=True, cbar=False, ax=ax, annot_kws={"size": 36}, cmap="Blues", vmin=df.values.min(), fmt='.3f')
    plt.xticks(rotation=0, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    plt.xlabel('Data', fontsize=25)
    plt.ylabel('Demographic Factors', fontsize=25)
    #plt.title(title, fontsize=25)
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()


topn = 500
if not os.path.exists('results_' + str(topn) + '.json'):
    data_list = [
        'twitter',
        'amazon',
        'yelp_hotel',
        'yelp_rest',
    ]
    
    results = dict()
    for data_name in data_list:
        results[data_name] = cal_overlap(data_name, topn)
    print(results)
    json.dump(results, open('results_' + str(topn) + '.json', 'w'))

from collections import OrderedDict
test = OrderedDict(json.load(open('results_' + str(topn) + '.json')))
df = pd.DataFrame(test).transpose()
df = df[['Gender', 'Age', 'Country', 'Region']]
viz_perform(df.transpose()) 
