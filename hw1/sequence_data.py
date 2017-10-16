import utils
import os
import pandas as pd
import numpy as np

datadir='data/'
feature='fbank'

xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                              header=None, names=['id', 'label'])

merged = xtrain.merge(y_train, how='left')
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])

sents = []
labels = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):
        label = np.array([i for i in fea['label'].values])
        label = label.reshape(label.shape[0], 1)
        lpadding = np.array(['sil' for i in range(777-label.shape[0])])
        lpadding = lpadding.reshape((lpadding.shape[0], 1))
        
        frames = np.array([i for i in fea['feature'].values])
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
        labels.append(np.append(label, lpadding, axis=0))

new = np.array([i for i in sents])
np.save('./data/{}/sents'.format(feature), new)

merged = utils.load_data(os.path.join(datadir, feature), flag='test')
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])
sents = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):       
        frames = np.array([i for i in fea['feature'].values])
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
new = np.array([i for i in sents])
np.save('./data/{}/test_sents'.format(feature), new)


feature='mfcc'
xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                              header=None, names=['id', 'label'])

merged = xtrain.merge(y_train, how='left')
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])

sents = []
labels = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):
        label = np.array([i for i in fea['label'].values])
        label = label.reshape(label.shape[0], 1)
        lpadding = np.array(['sil' for i in range(777-label.shape[0])])
        lpadding = lpadding.reshape((lpadding.shape[0], 1))
        
        frames = np.array([i for i in fea['feature'].values])
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
        labels.append(np.append(label, lpadding, axis=0))

new = np.array([i for i in sents])
np.save('./data/{}/sents'.format(feature), new)

merged = utils.load_data(os.path.join(datadir, feature), flag='test')
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])
sents = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):       
        frames = np.array([i for i in fea['feature'].values])
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
new = np.array([i for i in sents])
np.save('./data/{}/test_sents'.format(feature), new)



