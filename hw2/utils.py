import pandas as pd
import numpy as np
import os, sys, json, pickle
import glob
from keras.preprocessing import text

def load_data(path=os.path.join('.', 'data'), flag='train'):
    '''
    Return (training_data, training_label) or (testing_data, testing_label)
    training_data: (n_samples, n_frames, n_features)
    training_label: [list of (max_cap_len)] 
    '''
    if flag != 'train' and flag != 'test':
        raise 'Usage: load_data(path, flag=train or test)'

    # Load in data and label
    with open(os.path.join(path, '{}ing_label.json'.format(flag)), 'r') as j:
        label = json.load(j)
        path = os.path.join(path, '{}ing_data'.format(flag), 'feat')

    paths = sorted(glob.glob(os.path.join(path, '*.avi.npy')), key=lambda x: x[0])

    x_train = []
    y_train = []
    for p in paths:       
        x_train.append(np.load(p))
        idx = p.split('/')[-1][:-4]
        ans = next((item for item in label if item['id'] == idx), None)
        y_train.append(max(ans['caption'], key=len)) # choose the longest caption as training label

    x_train = np.array(x_train)

    np.save('./data/{}'.format(flag), x_train)
    return x_train, y_train

def preprocess_caps(train_caps, test_caps):
    '''
    Wrapper of caption preprocessing
    '''
    caps = train_caps + test_caps
    caps += ['<bos>', '<eos>', '<pad>', '<unk>']



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

if __name__ == "__main__":
    load_data(new=True)
