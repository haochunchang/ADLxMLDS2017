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

def get_sequence(merged, steps, feature):
    '''
    Preprocess features, avoiding cross-sentences steps
    For each sentence start, pad with zeros, not previous sentence

    merged: DataFrame with id, labels and features
    steps: timesteps considered
    Return: (n_samples, steps, n_features) 
    ''' 
    merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']]) 
    merged = merged.drop(['id', 'label'], axis=1)

    padding = np.zeros((steps // 2, len(merged['feature'].values[0])))
    i = 0
    for person, new_df in merged.groupby(level=0):
        x_train = np.empty((1, steps, len(merged['feature'].values[0])))
        for sentence, fea in new_df.groupby(level=1):
            frames = np.append(padding, np.array([i for i in fea['feature'].values]), axis=0)
            frames = np.append(frames, padding, axis=0)
            x_train = np.append(x_train, np.array([frames[i-steps//2:i+steps//2, :] 
                                                   for i in range(steps//2, frames.shape[0]-steps//2)]), axis=0)
        x_train = np.delete(x_train, (0), axis=0)
        print(i, x_train.shape)
        np.save('./data/{}/{}_steps{}_{}'.format(feature, feature, steps, i), x_train)
        i += 1

def combine_data(feature, steps):
    paths = glob.glob(os.path.join('data', feature, '{}_steps{}*'.format(feature, steps)))
    if feature == 'fbank':
        x_train = np.empty((1, steps, 69))
    else:
        x_train = np.empty((1, steps, 39))

    for i in range(462):
        path = os.path.join('data', feature, '{}_steps{}_{}.npy'.format(feature, steps, i))
        tmp = np.load(path)
        print('{} loaded'.format(path))
        x_train = np.append(x_train, tmp, axis=0)
    x_train = np.delete(x_train, (0), axis=0)     
    print(x_train.shape)
    np.save('{}_all_steps{}'.format(feature, steps), x_train)

if __name__ == "__main__":
    combine_data('fbank', 20)
    combine_data('mfcc', 20)
