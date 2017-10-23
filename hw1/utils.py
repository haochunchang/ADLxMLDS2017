import pandas as pd
import numpy as np
import os
import glob

def load_data(path, flag='train'):

    idx = []
    feature = []
    with open(os.path.join(path, '{}.ark'.format(flag)), 'r') as f:
        for line in f:
            idx.append(line.split(' ')[0])
            feature.append(np.array(line.split(' ')[1:], dtype=np.float32))
    
    data = pd.DataFrame()
    data['id'] = idx
    data['feature'] = feature
    return data

def phone_to_char(result, datadir):
    
    phone_char = pd.read_csv(os.path.join(datadir, '48phone_char.map'), sep='\t', 
                             header=None, index_col=0, names=['num', 'char'])['char']
    
    map39 = pd.read_csv(os.path.join(datadir, 'phones', '48_39.map'), sep='\t', 
                        header=None, index_col=0, names=['39phones'])
    
    chars = pd.Series()
    for x in range(result.shape[0]):
        result39 = [map39.loc[i].values[0] for i in result.iloc[x].split(',')]
        chars[result.index[x]] = [phone_char.loc[i][0] for i in result39]
    return chars

def trim(result, datadir):
    from itertools import groupby
    
    # convert phone to characters
    result = phone_to_char(result, datadir)
    
    # remove consecutive dupicates
    for i in range(result.shape[0]):
        new_serie = [k for k, g in groupby(result.iloc[i])]
        result.iloc[i] = new_serie
        
    # remove trailing and leading <sil>
    for i in range(result.shape[0]):
        for loc in (0, -1):
            while result.iloc[i] and result.iloc[i][loc] == 'L':
                result.iloc[i].pop(loc)  
    result = result.apply(lambda x: ''.join(x)) 
    return result

def combine_phone_seq(res):
    
    new_idx = res['id'].str.split('_').str[:2].str.join('_')

    new_result = res.groupby(new_idx).apply(lambda x: x['pred'].str.cat(sep=','))
    return new_result

def get_test_sequence(fbank, mfcc, save_all=True):
    '''
    Preprocess features, avoiding cross-sentences steps
    For each sentence start, pad with zeros, not previous sentence

    merged: DataFrame with id, labels and features
    steps: timesteps considered
    Return: (n_samples, steps, n_features) 
    ''' 
    fbank.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in fbank['id']])
    sents = []
    for person, new_df in fbank.groupby(level=0):
        for sentence, fea in new_df.groupby(level=1):       
            frames = np.array([i for i in fea['feature'].values])
            padding = np.zeros((777-frames.shape[0], frames.shape[1]))
            sents.append(np.append(frames, padding, axis=0))
    new_f = np.array([i for i in sents])
    if save_all:
        np.save('./fbank/test_sents', new_f)

    mfcc.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in mfcc['id']])
    sents = []
    for person, new_df in mfcc.groupby(level=0):
    	for sentence, fea in new_df.groupby(level=1):       
            frames = np.array([i for i in fea['feature'].values])
            padding = np.zeros((777-frames.shape[0], frames.shape[1]))
            sents.append(np.append(frames, padding, axis=0))

    new_m = np.array([i for i in sents])
    if save_all:
        np.save('./mfcc/test_sents', new_m)      
    new = np.append(new_f, new_m, axis=2)
    np.save('./test_sents', new)
    print("Test data preprocessing done.")


if __name__ == "__main__":
    result = pd.read_csv('prime_result.csv', index_col=0)
    result = combine_phone_seq(result)
    result = trim(result, './data/')
    sub = pd.DataFrame()
    sub['id'] = result.index
    sub['phone_sequence'] = result.values
    sub.to_csv('~/Desktop/sub.csv', index=False)  
 
