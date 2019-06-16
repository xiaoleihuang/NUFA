from utils import model_helper
import pickle
import os
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=33)

data_dirs = [
    ('../../data/twitter', 'twitter'),
    ('../../data/amazon', 'amazon'),
    ('../../data/yelp_hotel', 'yelp_hotel'),
    ('../../data/yelp_rest', 'yelp_rest'),
]

params_list = {
        'amazon_month' : {'C': 3.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'amazon_year' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'twitter' : {'C': 3, 'l1_ratio': 0, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 300},
        'economy_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100},
        'parties_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100},
        'vaccine_month' : {'C': 1, 'l1_ratio': 0, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 50},
        'vaccine_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 300},
        'yelp_hotel_month' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'yelp_hotel_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'yelp_rest_month' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'yelp_rest_year' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 300},
        'dianping_year': {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'dianping_month' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
}

results = open('./results.txt', 'a')

if not os.path.exists('./clfs/'):
    os.mkdir('./clfs/')
if not os.path.exists('./vects/'):
    os.mkdir('./vects/')

for data_dir in data_dirs:
    print('Working on: ' + data_dir[0])    
    
    if not os.path.exists('./clfs/'+data_dir[1]+'.model'):      
        print('Building DA vectorizer.................')
        if not os.path.exists('./vects/'+data_dir[1]+'.pkl'):
            print('Loading all dataset for building vectorizer.............')
            # load the data
            dataset = []
            for tmp_file in ['.train', '.dev', '.test']:
                with open(data_dir[0] + '/'+ data_dir[1] +tmp_file) as data_file:
                    data_file.readline() # skip the 1st line
                    for line in data_file:
                        dataset.append(line.strip().replace('\n', ' ').split('\t')[1:])
#                if tmp_file == 'train.tsv':
#                    # training data size: 200000
#                    if len(dataset) > 200000:
#                        np.random.seed(33)
#                        indices = list(range(len(dataset)))
#                        np.random.shuffle(indices)
#                        indices = indices[:200000]
#                        dataset = [dataset[idx_tmp] for idx_tmp in indices]

            da_vect = model_helper.DomainVectorizer_tfidf(-5)
            da_vect.fit(dataset)
            del dataset
            print('Save the vectorizer..............')
            pickle.dump(da_vect, open('./vects/'+data_dir[1]+'.pkl', 'wb'))
        else:
            print('Loading vectorizer..........')
            da_vect = pickle.load(open('./vects/'+data_dir[1]+'.pkl', 'rb'))

        print('Loading training data.............')
        train_data = []
        train_label = []
        with open(data_dir[0] + '/'+ data_dir[1] + '.train') as data_file:
            data_file.readline()
            for line in data_file:
                infos = line.strip().split('\t')
                train_data.append(infos)
                train_label.append(int(infos[-1]))
        # balance the data
        sample_indices = np.asarray([[item] for item in range(len(train_label))])
        sample_indices, train_label = ros.fit_sample(sample_indices, train_label)
        train_data = [train_data[item[0]] for item in sample_indices]
        del sample_indices
        
        # training data size: 200000
        if len(train_data) > 200000:
            np.random.seed(33)
            indices = list(range(len(train_data)))
            np.random.shuffle(indices)
            indices = indices[:200000]
            train_data = [train_data[idx_tmp] for idx_tmp in indices]
            train_label = [train_label[idx_tmp] for idx_tmp in indices]
        

        print('Transforming training data...........')
        train_data = da_vect.transform(train_data)
        
        print('Building classifier')
        clf = model_helper.build_lr_clf() # params_list[data_dir[1]]
        clf.fit(train_data, train_label)
        del train_data
        del train_label
        pickle.dump(clf, open('./clfs/'+data_dir[1]+'.pkl', 'wb'))
    else:
        # load clfs
        clf = pickle.load(open('./clfs/'+data_dir[1]+'.pkl', 'rb'))
    
    general_len = -1 * len(da_vect.tfidf_vec_da['general'].vocabulary_)
    best_lambda = 1
    best_valid = 0
    lambda_list = [0.3, 1, 10, 30, 100, 300]
    print('Loading Valid data')
    valid_data = []
    valid_label = []
    with open(data_dir[0] + '/'+ data_dir[1] + '.dev') as valid_file:
        valid_file.readline()
        for line in valid_file:
            infos = line.strip().split('\t')
            valid_data.append(infos)
            valid_label.append(int(infos[-1]))
    print('Transforming valid data....................')
    valid_data = da_vect.transform_test(valid_data)
    # for using only general features
    
    valid_data = sparse.lil_matrix(valid_data)
    # because the general features were appended finally, previous features are all domain features.
    valid_data[:, :general_len] = 0

    for lambda_item in lambda_list:
        exp_data = valid_data * lambda_item
        pred_label = clf.predict(exp_data)
        report_da = f1_score(y_true=valid_label, y_pred=pred_label, average='weighted')
        if report_da > best_valid:
            best_valid = report_da
            best_lambda = lambda_item

    # release memory
    del exp_data
    del valid_data
    del valid_label

    # load test
    print('Loading Test data')
    test_data = []
    test_label = []
    with open(data_dir[0] + '/' + data_dir[1] + '.test') as test_file:
        test_file.readline()
        for line in test_file:
            infos = line.strip().split('\t')
            test_data.append(infos)
            test_label.append(int(infos[-1]))


    print('Transforming test data....................')
    test_data = da_vect.transform_test(test_data)
    # for using only general features
    test_data = sparse.lil_matrix(test_data)
    test_data[:, :general_len] = 0
    test_data = test_data * best_lambda
    
    print('Testing.............................')
    pred_label = clf.predict(test_data)
    report_da = f1_score(y_true=test_label, y_pred=pred_label, average='weighted')

    results.write(data_dir[1] + ':' + str(report_da)+ '\n')
    results.write(classification_report(y_true=test_label, y_pred=pred_label, digits=3))
    results.write('...............................\n')
    results.flush()

results.close()
