import utils
import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer    

datadir='data/'
feature='fbank'
xtrainf = utils.load_data(os.path.join(datadir, '{}'.format(feature)))
y_train = pd.read_csv(os.path.join(datadir, 'label', 'train.lab'), header=None, names=['id', 'label'])

feature='mfcc'
xtrainm = utils.load_data(os.path.join(datadir, '{}'.format(feature)))

#print(xtrainf.head(), xtrainm.head(), y_train.head())
merged = xtrainf.merge(xtrainm, on='id', suffixes=('_f', '_m')).merge(y_train, on='id')
#merged = xtrainf.join([xtrainm, y_train], on='id', how='left', lsuffix='_f', rsuffix='_m')
print(merged.head())
merged.index = pd.MultiIndex.from_tuples([tuple(k.split('_')) for k in merged['id']])
print(merged.head())

# Save labelBinarizer
lb = LabelBinarizer()
lb.fit(merged['label'].values)
sil_loc = lb.transform(np.array(['sil'])).argmax()
print(sil_loc)
with open('label_map.pkl', 'wb') as f:
    pickle.dump(lb, f)
 

sents = []
labels = []
sent_end = []
for person, new_df in merged.groupby(level=0):
    for sentence, fea in new_df.groupby(level=1):
 
        label = np.array([i for i in fea['label'].values]) 
        label = lb.transform(label)
        lpadding = np.zeros((777-label.shape[0], label.shape[1]))
        lpadding[:, sil_loc] = 1
       
        fframes = np.array([i for i in fea['feature_f'].values])
        frames = np.append(fframes, np.array([i for i in fea['feature_m'].values]), axis=1)
        padding = np.zeros((777-frames.shape[0], frames.shape[1]))
        sents.append(np.append(frames, padding, axis=0))
        labels.append(np.append(label, lpadding, axis=0))
#       sent_end.append(frames.shape[0])

 

#np.save('./data/{}/sent_ends'.format(feature), np.array(sent_end))
np.save('./data/sents_labels', np.array([i for i in labels]))

new = np.array([i for i in sents])
np.save('./data/sents', new)

#======================================================================

