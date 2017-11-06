import pandas as pd
import numpy as np
import sys, json, pickle
from os import listdir
from os.path import join, isfile

def load_data(path=join('.', 'data'), flag='train'):
    '''
    Return (training_data, training_label) or (testing_data, testing_label)
    training_data: (n_samples, n_frames, n_features)
    training_label: [list of (max_cap_len)] 
    '''
    if flag != 'train' and flag != 'test':
        raise 'Usage: load_data(path, flag=train or test)'

    # Load in data and label
    with open(join(path, '{}ing_label.json'.format(flag)), 'r') as j:
        label = json.load(j)
    video_path = join(path, '{}ing_data'.format(flag), 'feat')
    paths = [f for f in listdir(video_path) if isfile(join(video_path, f))]
    
    x_train = []
    y_train = []
    index_lst = []
    for p in paths:
        x_train.append(np.load(join(video_path, p)))
        idx = p.split('/')[-1][:-4]
        index_lst.append(idx)
        ans = next((item for item in label if item['id'] == idx), None)
        y_train.append('<bos> '+max(ans['caption'], key=len)+' <eos>') # choose the longest caption as training label

    x_train = np.array(x_train)
    np.save(join(path, 'x_{}'.format(flag)), x_train)
    with open(join(path, 'y_{}.pkl'.format(flag)), 'wb') as p:
        pickle.dump((y_train, index_lst), p)

    return x_train, y_train     

def preprocess_caps(train_caps, test_caps, word_count_threshold):
    '''
    Wrapper of caption preprocessing
    '''
    captions = train_caps + test_caps
    captions = np.asarray(captions, dtype=np.object)
    
    # Filter out some common symbols in sentences
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    caps = captions
    return preBuildWordVocab(caps, word_count_threshold)

def preBuildWordVocab(sentence_iterator, word_count_threshold=5):
    '''
    Borrowed this function from NeuralTalk
    Build word vocabulary by counting words and filter words with count threshold
    Return: wordtoix dict, ixtoword dict, bias_init_vec according to word count
    '''
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
            vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    np.save("./data/wordtoix", wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save("./data/bias_init_vector", bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector

def pad_sequences(seq, maxlen=None, value=0.0):
    '''
    Pad seq with value to maxlen, truncate if exceed maxlen
    Expected input: list of lists of caption word index
    Return: np array with shape (n_samples, maxlen)
    '''
    if not hasattr(seq, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in seq:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                    'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    n_samples = len(seq)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in seq:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    # Initialize padded array with 1 * value
    x = (np.ones((n_samples, maxlen) + sample_shape) * value).astype(int)
    for idx, s in enumerate(seq):
        if not len(s):
            continue  # empty list/array was found
        trunc = s[:maxlen]

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=int)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                (trunc.shape[1:], idx, sample_shape))

        # Padding
        x[idx, :len(trunc)] = trunc
    
    return x

if __name__ == "__main__":
    datadir='./data'
    corpus = load_data(datadir, flag='train')
    print(corpus)
