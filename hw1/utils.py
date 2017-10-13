import pandas as pd
import numpy as np
import os

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
        result39 = [map39.loc[i].values[0] for i in result.iloc[x]]
        chars[result.index[x]] = ''.join([phone_char.loc[i][0] for i in result39])
    return chars

def trim(result, datadir):
    from itertools import groupby

    # remove consecutive dupicates
    for i in range(result.shape[0]):
        new_serie = [k for k, g in groupby(result.iloc[i].split(','))]
        result.iloc[i] = new_serie

    # remove trailing and leading <sil>
    for i in range(result.shape[0]):
        for loc in (0, -1):
            while result.iloc[i] and result.iloc[i][loc] == 'sil':
                result.iloc[i].pop(loc)  
 
    # convert phone to characters
    result = phone_to_char(result, datadir)
    
    return result

def combine_phone_seq(res):
    
    new_idx = res['id'].str.split('_').str[:2].str.join('_')

    new_result = res.groupby(new_idx).apply(lambda x: x['pred'].str.cat(sep=','))
    return new_result

def get_sequence(merged, steps):
    '''
    Preprocess features, avoiding cross-sentences steps
    For each sentence start, pad with zeros, not previous sentence

    merged: DataFrame with id, labels and features
    steps: timesteps considered
    Return: (n_samples, steps, n_features) 
    ''' 
    merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']]) 
    merged = merged.drop(['id', 'label'], axis=1)

    padding = np.array([[0 for i in range(len(merged['feature'].values[0]))] for j in range(steps)])
    x_train = np.empty([1, steps, len(merged['feature'].values[0])])
    print('Initialize empty x_train: {}'.format(x_train.shape))
    for person, new_df in merged.groupby(level=0):
        for sentence, fea in new_df.groupby(level=1):
            frames = np.append(padding, np.array([i for i in fea['feature'].values]), axis=0)
            x_train = np.append(x_train, np.array([frames[i:i+steps, :] for i in range(frames.shape[0]-steps)]), axis=0)
    x_train = np.delete(x_train, 0, axis=0)
    return x_train

if __name__ == "__main__":
    path = os.path.join('./data')
    test_result = pd.read_csv('test_result.csv')
    trim(test_result, path)
