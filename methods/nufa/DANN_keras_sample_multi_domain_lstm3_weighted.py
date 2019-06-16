"""
    Sample data from each single domain, create one input for each domain,

    3 BiLSTM, one for shared, one for domain, one for classification
"""

from keras.layers import Input, Conv1D, Embedding, Dropout, LSTM, Bidirectional
from keras.layers import MaxPool1D, Dense, Flatten
from keras.models import Model
from utils_dann import flipGradientTF
import numpy as np
from sklearn.metrics.classification import f1_score
from sklearn.metrics import classification_report
import sklearn
import keras
# original paper: https://arxiv.org/pdf/1505.07818.pdf
# model reference: https://cloud.githubusercontent.com/assets/7519133/19722698/9d1851fc-9bc3-11e6-96af-c2c845786f28.png


# force to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# load domain data
def load_domain_iter(data_name, dkeys, suffix='train'):
    """for single domain
        dkeys: the index of target domain
    """
    datap = '../../data_indices/' + data_name + '/' + data_name + '.' + suffix
    domain_docs = dict()
    dlabel_encoder = dict()

    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            
            # split the indices of doc
            line[1] = [int(item) for item in line[1].split()]

            # add domain label
            for dkey in dkeys:
                if line[dkey] == 'x':
                    continue

                line[dkey] = int(line[dkey])

                if dkey not in dlabel_encoder:
                    dlabel_encoder[dkey] = set()
                dlabel_encoder[dkey].add(line[dkey])

                if dkey not in domain_docs:
                    domain_docs[dkey] = []
                domain_docs[dkey].append(line)

    # find out the domain labels are not binary
    onehots = [dkey for dkey in dlabel_encoder.keys() if len(dlabel_encoder[dkey]) > 2]
    for dkey in dlabel_encoder:
        dlabel_encoder[dkey] = sorted(list(dlabel_encoder[dkey]))

    # encode domain label
    for dkey in onehots:
        for idx in range(len(domain_docs[dkey])):
            dl = [0] * len(dlabel_encoder[dkey])
            dl[dlabel_encoder[dkey].index(domain_docs[dkey][idx][dkey])] = 1
            domain_docs[dkey][idx][dkey] = dl

    return domain_docs, dlabel_encoder


def domain_data_gen(domain_docs, dkeys, batch_size=64):
    """
        Batch generator, for singel domain
    """
    batch_docs = dict()
    batch_labels = dict()

    for dkey in domain_docs:
        tmp_docs = np.random.choice(list(range(len(domain_docs[dkey]))), size=batch_size, replace=False)
        tmp_docs = [domain_docs[dkey][idx] for idx in tmp_docs]
        doc_key = 'domain'+str(dkey)+'_input'
        label_key = 'domain'+str(dkey)+'_pred'
        batch_docs[doc_key] = []
        batch_labels[label_key] = []

        for tmp_doc in tmp_docs:
            batch_docs[doc_key].append(tmp_doc[1])
            batch_labels[label_key].append(tmp_doc[dkey])

    # convert to array
    for doc_key in batch_docs:
        batch_docs[doc_key] = np.asarray(batch_docs[doc_key])
    for label_key in batch_labels:
        batch_labels[label_key] = np.asarray(batch_labels[label_key])

    return batch_docs, batch_labels


# load data
def load_data_iter(data_name, suffix='train'):
    """
        dkeys: the index of domain, list format
    """
    datap = '../../data_indices/' + data_name + '/' + data_name + '.' + suffix
    docs = []
    dlabel_encoder = dict()

    # filter flag, only for training, to filter out domain label 'x'
    flag = False

    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            
            # split the indices of doc
            line[1] = [int(item) for item in line[1].split()]

            docs.append(line)

    return docs


def data_gen(docs, batch_size=64):
    """
        Batch generator
    """
    np.random.shuffle(docs) # random shuffle the training documents
    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_docs = []
        batch_labels = []

        for idx in range(step*batch_size, (step+1)*batch_size):
            if idx > len(docs) -1:
                break
            batch_docs.append(np.asarray(docs[idx][1]))
            batch_labels.append(int(docs[idx][-1]))

        # convert to array
        batch_docs = np.asarray(batch_docs)
        batch_labels = np.asarray(batch_labels)

        yield batch_docs, batch_labels


def run_dnn(data_name, dkeys, domain_weights):
    print('Working on: ' + data_name)
    # parameters
    sent_len = 50 # the max length of sentence
    wt_path = '../../data/weight/'+ data_name + '.npy'
    epoch_num = 10
    dp_rate = 0.2
    # optimizer
    optimizer = keras.optimizers.Adam(0.0003)

    """Preprocess"""
    # training data
    train_data = load_data_iter(
        data_name, suffix='train'
    )
    # domain data
    domain_data, dlabel_encoder = load_domain_iter(data_name, dkeys)

    # load weights
    weights = np.load(wt_path)

    """Model, share layers between domain inputs and sentiment inputs"""
    # embedding
    embedding = Embedding(
        weights.shape[0], weights.shape[1], # size of data embedding
        weights=[weights], input_length=sent_len,
        trainable=True,
        name='embedding'
    )
    
    # Bi-LSTM
    bilstm_shared = Bidirectional(LSTM(200, dropout=dp_rate), name='shared_lstm')
    bilstm_senti = Bidirectional(LSTM(200, dropout=dp_rate), name='senti_lstm')
    bilstm_domain = Bidirectional(LSTM(200, dropout=dp_rate), name='domain_lstm')

    '''for sentiment clfs'''
    # input
    text_input = Input(shape=(sent_len,), dtype='int32', name='text_input')
    # define sentiment task layers
    emb_senti = embedding(text_input)
    lstm_shared = bilstm_shared(emb_senti)
    lstm_senti = bilstm_senti(emb_senti)
    merge_lstm = keras.layers.concatenate([lstm_senti, lstm_shared], axis=-1)
    dense_1 = Dense(128, activation='relu')(merge_lstm)
    dense_dp = Dropout(dp_rate)(dense_1)
    senti_preds = Dense(1, activation='sigmoid', name='senti')(dense_dp) # binary

    # for domain prediction
    hp_lambda = 0.01

    """Obtain the number of domain label, share layers with sentiment task"""
    domain_inputs = dict()
    for dkey in dkeys:
        domain_inputs[dkey] = [
            Input(
                shape=(sent_len,), dtype='int32', name='domain'+str(dkey)+'_input'
            )
        ]
        # shared layers start
        domain_inputs[dkey].append(
            embedding(domain_inputs[dkey][-1])
        )
        domain_inputs[dkey].append(
            bilstm_shared(domain_inputs[dkey][-1])
        )
        # shared layers end
        domain_inputs[dkey].append(
            bilstm_domain(domain_inputs[dkey][-2]) # embedding as input
        )
        domain_inputs[dkey].append(
            keras.layers.concatenate([domain_inputs[dkey][-2], domain_inputs[dkey][-1]], axis=-1)
        )
        domain_inputs[dkey].append(
            flipGradientTF.GradientReversal(
                hp_lambda, name='domain'+str(dkey)+'_flip'
            )(domain_inputs[dkey][-1])
        )
#        domain_inputs[dkey].append(
#            Dense(
#                128, activation='relu', name='domain'+str(dkey)+'_dense'
#            )(domain_inputs[dkey][-1])
#        )
#        domain_inputs[dkey].append(Dropout(dp_rate)(domain_inputs[dkey][-1]))

        dim_size = len(dlabel_encoder[dkey])
        print(dim_size)
        if dim_size == 2:
            dim_size = 1
        # check the label size
        if dim_size == 1:
            domain_inputs[dkey].append(
                Dense(
                    dim_size, activation='sigmoid', name='domain'+str(dkey)+'_pred'
                )(domain_inputs[dkey][-1])
            )
        else:
            domain_inputs[dkey].append(
                Dense(
                    dim_size, activation='softmax', name='domain'+str(dkey)+'_pred'
                )(domain_inputs[dkey][-1])
            )

    model_sentiment = Model(
        # the first element of each domain task is the input layer
        inputs=[text_input] + [domain_inputs[dkey][0] for dkey in sorted(dkeys)], 
        # the last layer of each domain task is the prediction layer        
        outputs=[senti_preds] + [domain_inputs[dkey][-1] for dkey in sorted(dkeys)],
    )

    # build loss (weight) for each domain
    loss_dict = {'senti': 'binary_crossentropy'}
    loss_w_dict = {'senti': 1}
    for dkey in dkeys:
        loss_w_dict['domain'+str(dkey)+'_pred'] = domain_weights[dkey]#0.1/len(dkeys)
        if len(dlabel_encoder[dkey]) > 2:
            loss_dict['domain'+str(dkey)+'_pred'] = 'categorical_crossentropy'
        else:
            loss_dict['domain'+str(dkey)+'_pred'] = 'binary_crossentropy'

    model_sentiment.compile(
        loss=loss_dict,
        loss_weights=loss_w_dict,
        optimizer=optimizer)
    print(model_sentiment.summary())

    # fit the model
    cls_w = {'senti:': 'auto'}
    for dkey in dkeys:
        cls_w['domain'+str(dkey)+'_pred'] = 'auto'

    # load the development set
    dev_data = load_data_iter(data_name, suffix='dev')
    best_dev = 0
    # test data
    test_data = load_data_iter(data_name, suffix='test')

    for e in range(epoch_num):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        train_iter = data_gen(train_data)
        # train sentiment
        # train on batches
        for x_train, y_labels in train_iter:
            batch_docs, batch_labels = domain_data_gen(domain_data, dkeys, len(x_train))
            batch_docs['text_input'] = x_train
            batch_labels['senti'] = y_labels

            # skip only 1 class in the training data
            if len(np.unique(batch_labels['senti'])) == 1:
                continue

            # train sentiment model
            tmp_senti = model_sentiment.train_on_batch(
                batch_docs,
                batch_labels,
                class_weight=cls_w,
            )
            # calculate loss and accuracy
            loss += tmp_senti[0]
            loss_avg = loss / step
            if step % 40 == 0:
                print('Step: {}'.format(step))
                #print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')
            step += 1

        # validation process
        y_preds_dev = []
        y_devs = []
        dev_iter = data_gen(dev_data)

        for x_dev, y_dev in dev_iter:
            x_dev = np.asarray(x_dev)
            tmp_preds = model_sentiment.predict([x_dev for _ in range(len(dkeys) + 1)])
            for item_tmp in tmp_preds[0]:
                y_preds_dev.append(np.round(item_tmp[0]))
            for item_tmp in y_dev:
                y_devs.append(int(item_tmp))
        cur_dev = f1_score(y_true=y_devs, y_pred=y_preds_dev, average='weighted')

        # if we get better dev result, test
        if cur_dev > best_dev:
            best_dev = cur_dev

            test_iter = data_gen(test_data)
            y_preds = []
            y_tests = []

            for x_test, y_test in test_iter:
                x_test = np.asarray(x_test)
                tmp_preds = model_sentiment.predict([x_test for _ in range(len(dkeys) + 1)])
                for item_tmp in tmp_preds[0]:
                    y_preds.append(np.round(item_tmp[0]))
                for item_tmp in y_test:
                    y_tests.append(int(item_tmp))
            test_result = open('./DANN_keras_sample_single_domain_lstm3_weighted_' + str(dkeys) + '.txt', 'a')
            test_result.write(data_name + '\t' + ','.join(map(str, dkeys)) + '\t' + str(e) + '\n')
            test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
            test_result.write(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))
            test_result.write('...............................................................\n\n')
            test_result.flush()


if __name__ == '__main__':
    data_list = [
        ('twitter', {-5: 0.276, -4: 0.14, -3: 0.279, -2: 0.306}),
        ('amazon', {-5: 0.289, -4: 0.163, -3: 0.281, -2:0.266}),
        ('yelp_hotel', {-5: 0.31, -4:0.151, -3: 0.28, -2:0.258}),
        ('yelp_rest', {-5: 0.323, -4: 0.162, -3: 0.269, -2:0.246}),
    ]

    domain_list = [[-5, -4, -3, -2]]
    for dname, weights in data_list:
        for dkeys in domain_list:
            run_dnn(dname, dkeys, weights)
