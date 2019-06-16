from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import os
import pickle
from imblearn.over_sampling import RandomOverSampler


def run_lr_3gram(data_name, train_path, test_path):
    """

    :param data_name:
    :type data_name: str
    :param train_path: training path
    :type train_path: str
    :param test_path: testing file path
    :type test_path: str
    :return:
    """
    print('Working on: '+data_name)
    # check if the vectorizer and exists
    # build the vectorizer
    if not (os.path.exists('./vects/' + data_name + '.pkl') and os.path.exists('./clfs/' + data_name + '.pkl')):
        print('Loading Training data........')
        # load the training data
        train_docs = []
        train_labels = []
        with open(train_path) as train_file:
            train_file.readline() # skip the 1st column names
            for line in train_file:
                if len(line.strip()) < 5:
                    continue
                
                infos = line.strip().split('\t')
                train_labels.append(int(infos[-1]))
                train_docs.append(infos[1].strip())

        # if training samples larger than 200,000
        if len(train_docs) > 200000:
            indices = list(range(len(train_docs)))
            np.random.seed(33)  # for reproducibility
            np.random.shuffle(indices)
            indices = indices[:200000]

            # get the first 200,000 data
            train_docs = [train_docs[tmp] for tmp in indices]
            train_labels = [train_labels[tmp] for tmp in indices]

        print('Fiting Vectorizer.......')
        vect = TfidfVectorizer(ngram_range=(1,3), max_features=15000, min_df=2)
        vect.fit(train_docs)
        pickle.dump(vect, open('./vects/'+data_name+'.pkl', 'wb')) # save the vectorizer

        print('Transforming Training data........')
        train_docs = vect.transform(train_docs)

        # fit the model
        print('Building model............')
        clf = SGDClassifier(class_weight='balanced')
        clf.fit(train_docs, train_labels)
        pickle.dump(clf, open('./clfs/' + data_name + '.pkl', 'wb'))  # save the classifier
    else:
        vect = pickle.load(open('./vects/'+data_name+'.pkl', 'rb'))
        clf = pickle.load(open('./clfs/'+data_name+'.pkl', 'rb'))

    # load the test data
    test_docs = []
    test_labels = []
    with open(test_path) as test_file:
        test_file.readline()
        for line in test_file:
            if len(line.strip()) < 5:
                continue
            infos = line.strip().split('\t')
            test_labels.append(int(infos[-1]))
            test_docs.append(infos[1].strip())

    # transform the test data
    print('Testing.........')
    test_docs = vect.transform(test_docs)
    y_preds = clf.predict(test_docs)

    with open('results.txt', 'a') as writefile:
        writefile.write(data_name + '_________________\n')
        writefile.write(str(f1_score(y_pred=y_preds, y_true=test_labels, average='weighted'))+'\n')
        writefile.write(classification_report(y_pred=y_preds, y_true=test_labels, digits=3)+'\n')
        writefile.write('.........................\n')


if __name__ == '__main__':
    data_list = [
        'twitter',
        'amazon',
        'yelp_hotel',
        'yelp_rest',
    ]
    for data_name in data_list:
        train_path = '../../data/'+data_name+'/' + data_name+'.train'
        test_path = '../../data/'+data_name+'/' + data_name+'.test'
        run_lr_3gram(data_name, train_path, test_path)
