"""
    This function is to create embedding weights for each dataset
"""
import pickle
import numpy as np
import gensim


def load_google(vec_path):
    """
    Load word2vec trained by Google Word2vec
    :param vec_path:
    :return:
    """
    if vec_path.endswith('bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
    else:
        model = gensim.models.KeyedVectors.load(vec_path)
    for pair in zip(model.wv.index2word, model.wv.syn0):
        yield pair[0], pair[1]


def load_glove(vec_path):
    """
    Load vectors trained by GloVe
    :param vec_path:
    :return:
    """
    with open(vec_path) as vec_file:
        for line in vec_file:
            tmp = line.strip().split()
            word = tmp[0]
            vectors = np.asarray(tmp[1:], dtype='float32')
            yield word, vectors


def load_fast(vec_path):
    """
    Load vectors trained by Fasttext
    :param vec_path:
    :return: a generator
    """
    with open(vec_path) as vec_file:
        # skip the 1st meta information
        vec_file.readline()
        for line in vec_file:
            tmp = line.strip().split()
            word = tmp[0]
            vectors = np.asarray(tmp[1:], dtype='float32')
            yield word, vectors


def load_w2v(vec_path):
    if 'fasttext' in vec_path:
        return load_fast(vec_path)
    if 'glove' in vec_path:
        return load_glove(vec_path)
    return load_google(vec_path)



def build_wt(datap):
    datap, vec_path = datap
    tkn_path = '../tokenizer/' + datap + '.tkn'

    print('Working on: ', datap)
    # load tokenizer
    with open(tkn_path, 'rb') as tkn_file:
        tkn = pickle.load(tkn_file)

    # get the vector generator
    vec_generator = load_w2v(vec_path)
    tmp_w, tmp_v = next(vec_generator)
    print('Embedding size: ' + str(len(tmp_v)))

    embed_len = len(tkn.word_index)
    if embed_len > tkn.num_words:
        embed_len = tkn.num_words

    embedding_matrix = np.zeros((embed_len + 1, len(tmp_v)))

    # loop through each word vectors
    print('Build vectors..............')
    for word, vectors in vec_generator:
        if word in tkn.word_index:
            if tkn.word_index[word] < tkn.num_words:
                embedding_matrix[tkn.word_index[word]] = vectors

    # save the matrix to the dir
    np.save(datap+'.npy', embedding_matrix)


data_list = [
    ('twitter', '../../w2v/vaccine.bin'),
    ('amazon', '../../w2v/amazon.bin'),
    ('yelp_rest', '../../w2v/yelp_rest.bin'),
    ('yelp_hotel', '../../w2v/yelp_hotel.bin'),
]

for datap in data_list:
    build_wt(datap)
