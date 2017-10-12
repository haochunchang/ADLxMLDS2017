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

if __name__ == "__main__":
    path = os.path.join('./data')
    trim('', path)
