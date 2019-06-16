import pickle
import gensim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def count_topics(data_name):
    print('Working on: ' + data_name)
    # load models
    lda_model = gensim.models.LdaModel.load(data_name + '.model')
    lda_dict = pickle.load(open(data_name + '.dict', 'rb'))

    # define demographic attributes
    results = dict()
    results['gender'] = {'male': [0]*10, 'female': [0]*10}
    results['age'] = {'young': [0]*10, 'old': [0]*10}
    results['country'] = {'US': [0]*10, 'no-US': [0]*10}
    results['region'] = {'NE': [0]*10, 'MW': [0]*10, 'S': [0]*10, 'W': [0]*10}

    # load the data
    with open('../data/' + data_name + '/' + data_name + '.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            topic_tmp = lda_model[lda_dict.doc2bow(line[1].split())]
            topic_doc = np.asarray([item[1] for item in topic_tmp])
            topest_idx = topic_doc.argsort()[0]

            # country
            if line[-5] == '1':
                results['country']['US'][topest_idx] += 1
            elif line[-5] == '0':
                results['country']['no-US'][topest_idx] += 1

            # region
            if line[-4] == '0':
                results['region']['S'][topest_idx] += 1
            elif line[-4] == '1':
                results['region']['W'][topest_idx] += 1
            elif line[-4] == '2':
                results['region']['MW'][topest_idx] += 1
            elif line[-4] == '3':
                results['region']['NE'][topest_idx] += 1

            # age
            if line[-2] == '1':
                results['age']['young'][topest_idx] += 1
            elif line[-2] == '0':
                results['age']['old'][topest_idx] += 1

            # gender
            if line[-3] == '1':
                results['gender']['female'][topest_idx] += 1
            elif line[-3] == '0':
                results['gender']['male'][topest_idx] += 1

    # normalize
    for key in results:
        for dkey in results[key]:

#            results[key][dkey] = [item**2 for item in results[key][dkey]]
            sum_v = sum(results[key][dkey])
            results[key][dkey] = [item/sum_v for item in results[key][dkey]]
        
        # convert to pandas data-frame, save to file
        if not os.path.exists('./results/' + data_name):
            os.mkdir('./results/' + data_name)
        topic_dist = pd.DataFrame.from_dict(results[key])
        topic_dist = topic_dist.transpose()
        pickle.dump(
            topic_dist, 
            open('./results/' + data_name + '/' + key + '.pkl', 'wb')
        )

        print(topic_dist)

        # create images
#        topic_dist.plot.bar(
#            stacked=True, legend=False, colormap='Paired', 
#            fontsize=20, figsize=(7,7),
#        )
#        #plt.title('Topic Change Over Time Sessions', fontsize=20)
#        plt.ylabel('Topic Proportion', fontsize=20)
#        plt.xticks(
#            list(range(len(results[key]))), 
#            list(results[key].keys()), fontsize=20, rotation=0,
#        )
#        plt.xlabel(data_name.capitalize() + ' - ' + key.capitalize(), rotation=0, fontsize=20)
#        plt.savefig('./images/' + data_name + '_' + key + '.pdf')


def print_topics(data_name, topn=15):
    # load models
    print('Data: ', data_name)
    lda_model = gensim.models.LdaModel.load(data_name + '.model')
    print('\n'.join(map(str, lda_model.print_topics(num_topics=10, num_words=topn))))
    print('----------------------------------------------------------------------\n')


if not os.path.exists('./results/'):
    os.mkdir('./results/')

if not os.path.exists('./images/'):
    os.mkdir('./images/')

data_list = [
    'twitter',
    'amazon', 
    'yelp_hotel',
    'yelp_rest',
]
for data_name in data_list:
    count_topics(data_name)
    print_topics(data_name, topn=15)
