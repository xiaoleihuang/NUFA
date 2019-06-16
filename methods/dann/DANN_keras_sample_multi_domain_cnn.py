"""
    Sample data from all input domains, create one input for all input domains
"""

from keras.layers import Input, Conv1D, Embedding, Dropout, LSTM, Bidirectional
from keras.layers import MaxPool1D, Dense, Flatten
from keras.models import Model
from utils_dann import flipGradientTF
import numpy as np
from sklearn.metrics.classification import f1_score
from sklearn.metrics import classification_report
import sklearn
# original paper: https://arxiv.org/pdf/1505.07818.pdf
# model reference: https://cloud.githubusercontent.com/assets/7519133/19722698/9d1851fc-9bc3-11e6-96af-c2c845786f28.png


# load domain data
def load_domain_iter(data_name, dkey, suffix='train'):
    """for single domain
        dkey: the index of target domain
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

            # filter out domain label with 'x' for training data
            if suffix == 'train':
                for dkey in dkeys:
                    if line[dkey] == 'x':
                        flag = True
            if flag:
                flag = False
                continue
            
            # split the indices of doc
            line[1] = [int(item) for item in line[1].split()]

            # add domain label
            if suffix == 'train':
                for dkey in dkeys:
                    line[dkey] = int(line[dkey])

                    if dkey not in dlabel_encoder:
                        dlabel_encoder[dkey] = set()
                    dlabel_encoder[dkey].add(line[dkey])

            docs.append(line)

    if suffix == 'train':
        # find out the domain labels are not binary
        onehots = [dkey for dkey in dlabel_encoder.keys() if len(dlabel_encoder[dkey]) > 2]
        for dkey in dlabel_encoder:
            dlabel_encoder[dkey] = sorted(list(dlabel_encoder[dkey]))

        # encode domain label
        for idx in range(len(docs)):
            for dkey in onehots:
                dl = [0] * len(dlabel_encoder[dkey])
                dl[dlabel_encoder[dkey].index(docs[idx][dkey])] = 1
                docs[idx][dkey] = dl

    return docs, dlabel_encoder


def domain_data_gen(docs, dkey, batch_size=64):
    """
        Batch generator, for singel domain
    """
    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_docs = []
        batch_labels = {'senti':[],}

        for idx in range(step*batch_size, (step+1)*batch_size):
            if idx > len(docs) -1:
                break
            batch_docs.append(np.asarray(docs[idx][1]))

            batch_labels['senti'].append(int(docs[idx][-1]))

            # domain labels
            for dkey in dkeys:
                label_key = 'domain'+str(dkey)+'_pred'
                if label_key not in batch_labels:
                    batch_labels[label_key] = []
                batch_labels[label_key].append(docs[idx][dkey])

        # convert to array
        batch_docs = np.asarray(batch_docs)
        for label_key in batch_labels:
            batch_labels[label_key] = np.asarray(batch_labels[label_key])

        yield batch_docs, batch_labels


# load domain data
def load_domain_iter_multi(data_name, dkeys, suffix='train'):
    """
        dkeys: the index of domain, list format
    """
    # TODO NOT FINISHED YET
    datap = '../../data_indices/' + data_name + '/' + data_name + '.' + suffix
    docs = []
    dlabel_encoder = dict()

    # filter flag, only for training, to filter out domain label 'x'
    flag = False

    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')

            # filter out domain label with 'x' for training data
            if suffix == 'train':
                for dkey in dkeys:
                    if line[dkey] == 'x':
                        flag = True
            if flag:
                flag = False
                continue
            
            # split the indices of doc
            line[1] = [int(item) for item in line[1].split()]

            # add domain label
            if suffix == 'train':
                for dkey in dkeys:
                    line[dkey] = int(line[dkey])

                    if dkey not in dlabel_encoder:
                        dlabel_encoder[dkey] = set()
                    dlabel_encoder[dkey].add(line[dkey])

            docs.append(line)

    if suffix == 'train':
        # find out the domain labels are not binary
        onehots = [dkey for dkey in dlabel_encoder.keys() if len(dlabel_encoder[dkey]) > 2]
        for dkey in dlabel_encoder:
            dlabel_encoder[dkey] = sorted(list(dlabel_encoder[dkey]))

        # encode domain label
        for idx in range(len(docs)):
            for dkey in onehots:
                dl = [0] * len(dlabel_encoder[dkey])
                dl[dlabel_encoder[dkey].index(docs[idx][dkey])] = 1
                docs[idx][dkey] = dl

    return docs, dlabel_encoder


def domain_data_gen_multi(docs, dkeys, batch_size=64):
    """
        Batch generator 
    """
    # TODO NOT FINISHED YET
    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_docs = []
        batch_labels = {'senti':[],}

        for idx in range(step*batch_size, (step+1)*batch_size):
            if idx > len(docs) -1:
                break
            batch_docs.append(np.asarray(docs[idx][1]))

            batch_labels['senti'].append(int(docs[idx][-1]))

            # domain labels
            for dkey in dkeys:
                label_key = 'domain'+str(dkey)+'_pred'
                if label_key not in batch_labels:
                    batch_labels[label_key] = []
                batch_labels[label_key].append(docs[idx][dkey])

        # convert to array
        batch_docs = np.asarray(batch_docs)
        for label_key in batch_labels:
            batch_labels[label_key] = np.asarray(batch_labels[label_key])

        yield batch_docs, batch_labels


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


def run_dnn(data_name, dkeys):
    print('Working on: ' + data_name)
    # parameters
    sent_len = 50 # the max length of sentence
    wt_path = '../../data/weight/'+ data_name + '.npy'
    epoch_num = 5
    

    """Preprocess"""
    # training data
    train_data, dlabel_encoder = load_data_iter(
        data_name, dkeys, suffix='train'
    )
    # load weights
    weights = np.load(wt_path)

    """Model"""
    # input
    text_input = Input(shape=(sent_len,), dtype='int32', name='text_input')

    # embedding
    embedding = Embedding(
        weights.shape[0], weights.shape[1], # size of data embedding
        weights=[weights], input_length=sent_len,
        trainable=True,
        name='embedding'
    )(text_input)

    # CNN
    conv1 = Conv1D(
        filters=300,
        kernel_size=5,
        padding='valid',
        strides=1,
    )(embedding)
    conv2 = Conv1D(
        filters=200,
        kernel_size=7,
        padding='valid',
        strides=1,
    )(conv1)
    max_pool = MaxPool1D()(conv2)

    flatten = Flatten()(max_pool)

    # for sentiment clfs
    dense_1 = Dense(128, activation='relu')(flatten)
    dense_dp = Dropout(0.2)(dense_1)
    sentiment_preds = Dense(1, activation='sigmoid', name='senti')(dense_dp) # binary

    # for domain prediction
    hp_lambda = 0.01

    """Obtain the number of domain label"""
    domain_inputs = dict()
    for dkey in dkeys:
        domain_inputs[dkey] = []
        domain_inputs[dkey].append(
            flipGradientTF.GradientReversal(
                hp_lambda, name='domain'+str(dkey)+'_flip'
            )(flatten)
        )
        domain_inputs[dkey].append(
            Dense(
                128, activation='relu', name='domain'+str(dkey)+'_dense'
            )(domain_inputs[dkey][-1])
        )
        domain_inputs[dkey].append(Dropout(0.2)(domain_inputs[dkey][-1]))

        dim_size = len(dlabel_encoder[dkey])
        print(dim_size)
        if dim_size == 2:
            dim_size = 1
        # check the label size
        domain_inputs[dkey].append(
            Dense(
                dim_size, activation='softmax', name='domain'+str(dkey)+'_pred'
            )(domain_inputs[dkey][-1])
        )

    model_sentiment = Model(
        inputs=[text_input], 
        # the last layer of each domain task is the prediction layer        
        outputs=[sentiment_preds] + [domain_inputs[dkey][-1] for dkey in sorted(dkeys)],
    )

    # build loss (weight) for each domain
    loss_dict = {'senti': 'binary_crossentropy'}
    loss_w_dict = {'senti': 1}
    for dkey in dkeys:
        loss_w_dict['domain'+str(dkey)+'_pred'] = 0.1
        if len(dlabel_encoder[dkey]) > 2:
            loss_dict['domain'+str(dkey)+'_pred'] = 'categorical_crossentropy'
        else:
            loss_dict['domain'+str(dkey)+'_pred'] = 'binary_crossentropy'

    model_sentiment.compile(
        loss=loss_dict,
        loss_weights=loss_w_dict,
        optimizer='adam')
    print(model_sentiment.summary())

    # fit the model
    cls_w = {'senti:': 'auto'}
    for dkey in dkeys:
        cls_w['domain'+str(dkey)+'_pred'] = 'auto'

    for e in range(epoch_num):
        # shuffle the data
        np.random.shuffle(train_data)

        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        train_iter = data_gen(train_data, dkeys)
        # train sentiment
        # train on batches
        for x_train, train_labels in train_iter:
            # skip only 1 class in the training data
            if len(np.unique(train_labels['senti'])) == 1:
                continue

            # train sentiment model
            tmp_senti = model_sentiment.train_on_batch(
                x_train,
                train_labels,
                class_weight=cls_w,
            )
            # calculate loss and accuracy
            loss += tmp_senti[0]
            loss_avg = loss / step
            if step % 40 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}.'.format(loss_avg))
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

            # test
            test_data, _ = load_data_iter(
                data_name, dkeys, suffix='test'
            )
            test_iter = data_gen(test_data, dkeys)

            y_preds = []
            y_tests = []
            for x_test, y_test in test_iter:
                x_test = np.asarray(x_test)
                tmp_preds = model_sentiment.predict(x_test)
                for item_tmp in tmp_preds[0]:
                    y_preds.append(np.round(item_tmp[0]))
                for item_tmp in y_test['senti']:
                    y_tests.append(int(item_tmp))
            test_result = open('./results_dann_sample_domain.txt', 'a')
            test_result.write(data_name + '\t' + ','.join(map(str, dkeys)) + '\n')
            test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
            test_result.write(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))
            test_result.write('...............................................................\n\n')

if __name__ == '__main__':
    data_list = [
        'twitter',
        'amazon',
        'yelp_hotel',
        'yelp_rest'
    ]

    domain_list = [[-5], [-4], [-3], [-2]]

    for dname in data_list:
        for dkeys in domain_list:
            run_dnn(dname, dkeys)
