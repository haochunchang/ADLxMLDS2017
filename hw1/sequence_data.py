import utils
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer    


datadir='data/'
feature='fbank'
xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), header=None, names=['id', 'label'])

merged = xtrain.merge(y_train, how='left')
    
# Save labelBinarizer
lb = LabelBinarizer()
lb.fit(merged['label'].values)
with open('label_map.pkl', 'wb') as f:
    pickle.dump(lb, f)
 
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])

sents = []
labels = []
sent_end = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):
 
        label = np.array([i for i in fea['label'].values]) 
        label = lb.transform(label)
        lpadding = np.zeros((777-label.shape[0], label.shape[1]))
        
        frames = np.array([i for i in fea['feature'].values])
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
        labels.append(np.append(label, lpadding, axis=0))
        sent_end.append(frames.shape[0])

np.save('./data/{}/sent_ends'.format(feature), np.array(sent_end))
np.save('./data/{}/sents_labels'.format(feature), np.array([i for i in labels]))

new = np.array([i for i in sents])
np.save('./data/{}/sents'.format(feature), new)

#======================================================================
feature='mfcc'

xtrain = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), 
                              header=None, names=['id', 'label'])

merged = xtrain.merge(y_train, how='left')
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])

sents = []
labels = []
sent_end = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):
 
        label = np.array([i for i in fea['label'].values]) 
        label = lb.transform(label)
        lpadding = np.zeros((777-label.shape[0], label.shape[1]))
        
        frames = np.array([i for i in fea['feature'].values])
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
        labels.append(np.append(label, lpadding, axis=0))
        sent_end.append(frames.shape[0])

np.save('./data/{}/sent_ends'.format(feature), np.array(sent_end))

np.save('./data/{}/sents_labels'.format(feature), np.array([i for i in labels]))
new = np.array([i for i in sents])
np.save('./data/{}/sents'.format(feature), new)


