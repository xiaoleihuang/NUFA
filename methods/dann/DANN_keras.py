from keras.layers import Input, Conv1D, Embedding, Dropout
from keras.layers import MaxPool1D, Dense, Flatten
from keras.models import Model
from utils_dann import flipGradientTF
import numpy as np
from sklearn.metrics.classification import f1_score
from sklearn.metrics import classification_report
# original paper: https://arxiv.org/pdf/1505.07818.pdf
# model reference: https://cloud.githubusercontent.com/assets/7519133/19722698/9d1851fc-9bc3-11e6-96af-c2c845786f28.png

data_list = [
    ('vaccine', 'vaccine_year'),
    ('amazon', 'amazon_month'),
    ('amazon', 'amazon_year'),
    ('dianping', 'dianping_month'),
    ('dianping', 'dianping_year'),
    ('google', 'economy_month'),
    ('google', 'economy_year'),
    ('google', 'parties_year'),
    ('vaccine', 'vaccine_month'),
    ('yelp_hotel', 'yelp_hotel_month'),
    ('yelp_hotel', 'yelp_hotel_year'),
    ('yelp_rest', 'yelp_rest_month'),
    ('yelp_rest', 'yelp_rest_year'),
]

# load data
def load_data_iter(filename, batch_size=64, train=True):
    domain_labels = []  # for encoding domain labels
    labels = []
    docs = []

    with open(filename) as data_file:
        for line in data_file:
            infos = line.strip().split('\t')
            if train:
                domain_labels.append(int(infos[1]))  # domain label position
                labels.append(int(infos[0]))
                docs.append([int(item) for item in infos[2:]])
            else:
                labels.append(int(infos[0]))
                docs.append([int(item) for item in infos[1:]])

    if train:
        uniqs = sorted(set(domain_labels))  # for one-hot encoding

        # check if the number of training example exceeds 150000
        if len(docs) > 200000:
            indices = list(range(len(docs)))
            np.random.seed(33) # for reproducibility
            np.random.shuffle(indices)
            indices = indices[:200000]

            # get the first 150000 data
            docs = [docs[tmp] for tmp in indices]
            labels = [labels[tmp] for tmp in indices]
            domain_labels = [domain_labels[tmp] for tmp in indices]

    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for idx in range(steps):
        batch_data = np.asarray(docs[idx*batch_size: (idx+1)*batch_size])
        batch_label = np.asarray(labels[idx*batch_size: (idx+1)*batch_size])

        if train:
            batch_domain_label = []
            start = idx * batch_size
            if (idx + 1) * batch_size > len(docs):
                end = len(docs)
            else:
                end = (idx + 1) * batch_size

            for tmp in range(start, end):
                dl = [0] * len(uniqs)
                dl[uniqs.index(domain_labels[tmp])] = 1
                batch_domain_label.append(dl)

            yield batch_data, np.asarray(batch_domain_label), batch_label
        else:
            yield batch_data, batch_label


def load_data_iter_1(filename, batch_size=64, train=True):
    count = 0  # count data length
    tmp_labels = [] # for encoding domain labels


    with open(filename) as data_file:
        for line in data_file:
            count += 1
            if train:
                tmp_labels.append(line.strip().split('\t')[1]) # domain label position

    if train:
        uniqs = sorted(set(tmp_labels)) # for one-hot encoding

    steps = int(count / batch_size)
    if count % batch_size != 0:
        steps += 1

    with open(filename) as data_file:
        for _ in range(steps):
            data = []
            label = []
            domain_label = []

            for _ in range(batch_size):
                line = data_file.readline()
                if line is None or len(line.strip()) == 0:
                    break
                infos = line.strip().split('\t')
                label.append(infos[0])

                if train:
                    dl = [0]*len(uniqs)
                    dl[uniqs.index(infos[1])] = 1
                    domain_label.append(dl)
                    tmp = [int(item) for item in infos[2:]]
                    data.append(tmp)
                else:
                    tmp = [int(item) for item in infos[1:]]
                    data.append(tmp)

            if train:
                yield data, domain_label, label
            else:
                yield data, label


def run_dnn(data_pair):
    print('Working on: '+data_pair[1])
    wt_path = './weights/'+ data_pair[1] + '.npy'
    train_path = './data/'+ data_pair[1] + '_source.txt'
    test_path = './data/'+ data_pair[1] + '_target.txt'
    epoch_num = 8

    # parameters
    sent_len = 50 # the max length of sentence

    """Preprocess"""
    # load weights
    weights = np.load(wt_path)
    print(weights.shape)

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
    da_num = set()
    with open(train_path) as data_file:
        for line in data_file:
            da_num.add(line.strip().split('\t')[1]) # domain label position

    flip = flipGradientTF.GradientReversal(hp_lambda)(flatten)
    dense_da = Dense(128, activation='relu')(flip)
    dense_da_dp = Dropout(0.2)(dense_da)
    da_preds = Dense(len(da_num), activation='softmax', name='domain')(dense_da_dp) # multiple

    model_sentiment = Model(
        inputs=[text_input], outputs=[sentiment_preds, da_preds],
    )
    model_sentiment.compile(
        loss={'senti': 'binary_crossentropy', 'domain':'categorical_crossentropy'},
        loss_weights={'senti': 1, 'domain':0.31},
        optimizer='adam')
    print(model_sentiment.summary())
    # fit the model
    for e in range(epoch_num):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        train_iter = load_data_iter(train_path)
        # train sentiment
        # train on batches
        for x_train, time_labels, y_train in train_iter:
            # skip only 1 class in the training data
            if len(np.unique(y_train)) == 1:
                continue

            # train sentiment model
            tmp_senti = model_sentiment.train_on_batch(
                x_train,
                {'senti': y_train, 'domain': time_labels},
                class_weight={'senti:': 'auto', 'domain': 'auto'}
            )
            # calculate loss and accuracy
            loss += tmp_senti[0]
            loss_avg = loss / step
            if step % 40 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')
            step += 1

    
    # test
    test_iter = load_data_iter(test_path, train=False)
    y_preds = []
    y_tests = []
    for x_test, y_test in test_iter:
        x_test = np.asarray(x_test)
        tmp_preds = model_sentiment.predict(x_test)
        for item_tmp in tmp_preds[0]:
            y_preds.append(np.round(item_tmp[0]))
        for item_tmp in y_test:
            y_tests.append(int(item_tmp))
    test_result = open('./results_dann.txt', 'a')
    test_result.write(data_pair[1] + '\n')
    test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
    test_result.write(classification_report(y_true=y_tests, y_pred=y_preds))
    test_result.write('...............................................................\n\n')

if __name__ == '__main__':
    for data_pair in data_list:
        run_dnn(data_pair)
